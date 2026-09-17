# Enhanced Interactive GUI for Swallow Detection and Analysis
# with Robust Excel Ground Truth Handling and SPL Frequency Analysis

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import librosa
from scipy import signal, ndimage
from scipy.signal import find_peaks, butter, filtfilt, welch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd
import re
from pathlib import Path
import threading
import warnings
import json
from dataclasses import dataclass
from typing import Optional
warnings.filterwarnings('ignore')

@dataclass
class GroundTruth:
    """Ground truth data from Excel"""
    filename: str
    swallow_count: int
    group: str
    volume_ml: Optional[float] = None
    event_durations: Optional[list] = None

class ExcelGroundTruthManager:
    """Manages ground truth data from Excel files"""
    
    def __init__(self):
        self.ground_truth_data = {}
        self.matching_report = []
    
    def load_excel_file(self, excel_path):
        """Load ground truth from Excel file with handling for missing audio files"""
        try:
            # Load both sheets
            control_df = pd.read_excel(excel_path, sheet_name="Grupo de Controle")
            study_df = pd.read_excel(excel_path, sheet_name="Grupo de Estudo")
            
            loaded_count = 0
            skipped_count = 0
            
            # Helper function to convert string to float handling comma decimal separator
            def safe_float_convert(value):
                """Convert string to float, handling comma decimal separators"""
                if pd.isna(value):
                    return None
                try:
                    # Convert to string first in case it's already a number
                    str_value = str(value).strip()
                    # Replace comma with period for decimal separator
                    str_value = str_value.replace(',', '.')
                    return float(str_value)
                except (ValueError, TypeError):
                    return None
            
            # Helper function to safely convert to integer
            def safe_int_convert(value):
                """Convert value to integer safely"""
                if pd.isna(value):
                    return 0
                try:
                    # Convert to string first, then to int
                    str_value = str(value).strip()
                    # Handle cases where the value might be a float string
                    if ',' in str_value:
                        str_value = str_value.replace(',', '.')
                    return int(float(str_value))
                except (ValueError, TypeError):
                    return 0
            
            # Process control group
            for _, row in control_df.iterrows():
                if pd.notna(row['ID']):
                    raw_id = str(row['ID']).strip()
                    filename = self.normalize_filename(raw_id)
                    
                    # Check if this looks like a valid audio file entry
                    if self.should_skip_entry(raw_id):
                        skipped_count += 1
                        continue
                    
                    # Safely convert swallow count to integer
                    swallow_count = safe_int_convert(row.get('Nr de Goles', 0))
                    
                    if swallow_count > 0:
                        # Extract event durations with safe conversion
                        event_durations = []
                        for i in range(1, 9):  # Check up to 8 events
                            event_col = f'{i} evento'
                            if event_col in row and pd.notna(row[event_col]):
                                duration = safe_float_convert(row[event_col])
                                if duration is not None:
                                    event_durations.append(duration)
                        
                        self.ground_truth_data[filename] = GroundTruth(
                            filename=filename,
                            swallow_count=swallow_count,
                            group='Control',
                            volume_ml=self.extract_volume_from_filename(raw_id),
                            event_durations=event_durations if event_durations else None
                        )
                        loaded_count += 1
                    else:
                        skipped_count += 1
            
            # Process study group
            for _, row in study_df.iterrows():
                if pd.notna(row['ID']):
                    raw_id = str(row['ID']).strip()
                    filename = self.normalize_filename(raw_id)
                    
                    # Safely convert swallow count to integer
                    swallow_count = safe_int_convert(row.get('Nr de deglutições', 0))
                    
                    if swallow_count > 0:
                        # Extract event durations with safe conversion
                        event_durations = []
                        # Study group has columns like "1 evento", "2 evento", "3º EVENTO", "4ºEVENTO", "5ºEvento"
                        event_columns = ['1 evento', '2 evento', '3º EVENTO', '4ºEVENTO', '5ºEvento']
                        for event_col in event_columns:
                            if event_col in row and pd.notna(row[event_col]):
                                duration = safe_float_convert(row[event_col])
                                if duration is not None:
                                    event_durations.append(duration)
                        
                        self.ground_truth_data[filename] = GroundTruth(
                            filename=filename,
                            swallow_count=swallow_count,
                            group='Study',
                            volume_ml=self.extract_volume_from_filename(raw_id),
                            event_durations=event_durations if event_durations else None
                        )
                        loaded_count += 1
                    else:
                        skipped_count += 1
            
            message = f"Loaded {loaded_count} ground truth entries"
            if skipped_count > 0:
                message += f" (skipped {skipped_count} entries without audio/data)"
            
            return True, message
            
        except Exception as e:
            return False, f"Error loading Excel file: {str(e)}"
    
    def should_skip_entry(self, raw_id):
        """Check if an Excel entry should be skipped"""
        if re.match(r'^[A-Z]\d+$', raw_id):  # Pattern like F1, F2, etc.
            return True
        if not re.search(r'\d+ml', raw_id, re.IGNORECASE):
            return True
        if len(raw_id) < 5:
            return True
        return False
    
    def normalize_filename(self, filename):
        """Normalize filename for matching"""
        name = re.sub(r'\.(wav|WAV)$', '', filename)
        name = name.upper().replace(' ', '').replace('_', '').replace('-', '')
        return name
    
    def extract_volume_from_filename(self, filename):
        """Extract volume information from filename with robust pattern matching"""
        name_without_ext = re.sub(r'\.(wav|WAV)$', '', filename)
        
        # Look for volume pattern: 1-2 digits before ml/ML
        patterns = [
            r'(\d{1,2})ml\b',
            r'(\d{1,2})ML\b',
            r'(\d{1,2})ml$',
            r'(\d{1,2})ML$',
        ]
        
        volume_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, name_without_ext)
            for match in matches:
                volume = int(match)
                if 1 <= volume <= 100:
                    volume_matches.append(volume)
        
        if volume_matches:
            unique_volumes = sorted(set(volume_matches))
            preferred_volumes = [v for v in unique_volumes if 5 <= v <= 50]
            if preferred_volumes:
                return float(preferred_volumes[0])
            else:
                return float(unique_volumes[0])
        
        return None
    
    def get_ground_truth(self, audio_filename):
        """Get ground truth for audio file"""
        normalized = self.normalize_filename(audio_filename)
        
        # Direct match
        if normalized in self.ground_truth_data:
            self.matching_report.append((audio_filename, normalized, 'exact'))
            return self.ground_truth_data[normalized]
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for gt_filename in self.ground_truth_data.keys():
            common_chars = len(set(normalized) & set(gt_filename))
            total_chars = len(set(normalized) | set(gt_filename))
            score = common_chars / total_chars if total_chars > 0 else 0
            
            if score > best_score and score > 0.7:
                best_score = score
                best_match = gt_filename
        
        if best_match:
            self.matching_report.append((audio_filename, best_match, f'fuzzy_{best_score:.2f}'))
            return self.ground_truth_data[best_match]
        
        self.matching_report.append((audio_filename, None, 'no_match'))
        return None

class SwallowDetectorGUI:
    def __init__(self, sr=22050):
        self.sr = sr
        self.detection_params = {
            'min_swallow_duration': 0.1,
            'max_swallow_duration': 3.0,
            'min_event_separation': 0.5,
            'energy_percentile': 60,
            'adaptive_threshold': True,
            'min_prominence': 0.01,
            'smoothing_sigma': 2,
            'boundary_method': 'energy_based',
            'energy_drop_threshold': 0.3,
            'merge_close_events': True,
            'min_silence_duration': 0.1,
        }
        
        # GUI State
        self.root = None
        self.audio_files = []
        self.current_file_idx = 0
        self.current_data = None
        self.current_ground_truth = None
        self.audio_folder = None
        
        # Excel ground truth manager
        self.ground_truth_manager = ExcelGroundTruthManager()
        
        # Storage systems
        self.spl_analysis_storage = []
        self.file_parameter_storage = {}
        self.file_results_storage = {}
        
        # Matplotlib components
        self.fig = None
        self.canvas = None
        self.axes = None
        
        # Parameter widgets
        self.param_widgets = {}
        self.info_label = None
        self.ground_truth_label = None
        self.spl_storage_label = None
        self.param_storage_label = None
    
    def preprocess_stethoscope_audio(self, y, sr):
        nyquist = sr / 2
        low = 20 / nyquist
        high = min(2000 / nyquist, 0.99)
        
        b, a = butter(4, [low, high], btype='band')
        y_filtered = filtfilt(b, a, y)
        y_normalized = librosa.util.normalize(y_filtered)
        return y_normalized
    
    def detect_swallow_events(self, y, sr):
        """Detect swallow events with current parameters"""
        y_clean = self.preprocess_stethoscope_audio(y, sr)
        
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        rms = librosa.feature.rms(y=y_clean, frame_length=frame_length, hop_length=hop_length)[0]
        rms_smooth = ndimage.gaussian_filter1d(rms, sigma=self.detection_params['smoothing_sigma'])
        
        # Calculate threshold
        if self.detection_params['adaptive_threshold']:
            threshold_methods = [
                np.percentile(rms_smooth, self.detection_params['energy_percentile']),
                np.mean(rms_smooth) + 1.5 * np.std(rms_smooth),
                np.median(rms_smooth) + 1.0 * np.std(rms_smooth)
            ]
            threshold = min(threshold_methods)
        else:
            threshold = np.percentile(rms_smooth, self.detection_params['energy_percentile'])
        
        # Cap threshold
        max_allowed_threshold = np.max(rms_smooth) * 0.25
        if threshold > max_allowed_threshold:
            threshold = max_allowed_threshold
        
        # Find peaks
        min_distance_frames = int(0.1 * sr / hop_length)
        peaks, properties = find_peaks(
            rms_smooth,
            height=threshold,
            distance=min_distance_frames,
            prominence=self.detection_params['min_prominence']
        )
        
        # Create events
        events = []
        for peak_idx in peaks:
            peak_time = peak_idx * hop_length / sr
            
            # Find boundaries
            start_time, end_time = self.find_event_boundaries(
                rms_smooth, peak_idx, hop_length, sr, threshold)
            
            duration = end_time - start_time
            
            # Validate duration
            if (self.detection_params['min_swallow_duration'] <= 
                duration <= 
                self.detection_params['max_swallow_duration']):
                
                confidence = min(1.0, rms_smooth[peak_idx] / (threshold + 1e-10))
                
                events.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': duration,
                    'peak_time': peak_time,
                    'peak_energy': rms_smooth[peak_idx],
                    'confidence': confidence
                })
        
        # Merge close events if enabled
        if self.detection_params['merge_close_events']:
            events = self.merge_overlapping_events(events)
        
        return events, y_clean, rms_smooth, threshold
    
    def find_event_boundaries(self, rms_smooth, peak_idx, hop_length, sr, threshold):
        """Find event boundaries based on energy drop"""
        peak_energy = rms_smooth[peak_idx]
        min_energy_threshold = peak_energy * self.detection_params['energy_drop_threshold']
        
        search_window = int(2.0 * sr / hop_length)
        start_idx = max(0, peak_idx - search_window)
        end_idx = min(len(rms_smooth), peak_idx + search_window)
        
        # Find start boundary
        event_start = start_idx
        for i in range(peak_idx, start_idx, -1):
            if rms_smooth[i] < min_energy_threshold:
                event_start = i
                break
        
        # Find end boundary
        event_end = end_idx
        for i in range(peak_idx, end_idx):
            if rms_smooth[i] < min_energy_threshold:
                event_end = i
                break
        
        return (event_start * hop_length / sr, event_end * hop_length / sr)
    
    def merge_overlapping_events(self, events):
        """Merge events that are too close together"""
        if len(events) <= 1:
            return events
        
        events.sort(key=lambda x: x['start'])
        merged_events = []
        current_event = events[0].copy()
        
        for next_event in events[1:]:
            separation = next_event['start'] - current_event['end']
            
            if separation < self.detection_params['min_event_separation']:
                # Merge events
                current_event['end'] = max(current_event['end'], next_event['end'])
                current_event['duration'] = current_event['end'] - current_event['start']
                
                if next_event['peak_energy'] > current_event['peak_energy']:
                    current_event['peak_time'] = next_event['peak_time']
                    current_event['peak_energy'] = next_event['peak_energy']
                
                current_event['confidence'] = (current_event['confidence'] + next_event['confidence']) / 2
            else:
                merged_events.append(current_event)
                current_event = next_event.copy()
        
        merged_events.append(current_event)
        return merged_events
    
    def calculate_spl_vs_frequency_detailed(self, segment, sr, n_fft=2048, overlap_ratio=0.75):
        """Calculate detailed SPL for frequency analysis with higher resolution"""
        if len(segment) < n_fft:
            return None, None
        
        # Use higher resolution FFT analysis
        hop_length = int(n_fft * (1 - overlap_ratio))
        
        # Compute spectrogram with higher frequency resolution
        freqs, times, Sxx = signal.spectrogram(
            segment, 
            sr, 
            nperseg=n_fft, 
            noverlap=int(n_fft * overlap_ratio),
            window='hann'
        )
        
        # Calculate mean power across time for each frequency
        mean_power = np.mean(Sxx, axis=1)
        
        # Convert to SPL (dB re 20 μPa)
        # Using reference pressure of 20 μPa = 2e-5 Pa
        reference_pressure = 2e-5
        spl_values = 10 * np.log10(mean_power / (reference_pressure**2))
        
        # Filter to reasonable frequency range (20-2000 Hz to match bandpass filter)
        freq_mask = (freqs >= 20) & (freqs <= 2000)
        freqs_filtered = freqs[freq_mask]
        spl_filtered = spl_values[freq_mask]
        
        return freqs_filtered, spl_filtered
    
    def save_current_analysis_state(self):
        """Save current parameters and results for the current file"""
        if not self.current_data:
            return
        
        filename = self.current_data['filename']
        
        # Save parameters
        self.file_parameter_storage[filename] = self.detection_params.copy()
        
        # Save results
        self.file_results_storage[filename] = {
            'events': self.current_data['events'].copy() if self.current_data['events'] else [],
            'threshold': self.current_data['threshold'],
            'ground_truth': self.current_ground_truth,
            'timestamp': pd.Timestamp.now()
        }
        
        # Update display
        self.update_parameter_storage_info()
    
    def load_analysis_state_for_file(self, filename):
        """Load saved parameters and results for a specific file"""
        if filename in self.file_parameter_storage:
            # Load parameters
            saved_params = self.file_parameter_storage[filename]
            self.detection_params.update(saved_params)
            
            # Update GUI widgets to reflect loaded parameters
            for param, value in saved_params.items():
                if param in self.param_widgets:
                    if isinstance(self.param_widgets[param], tk.Scale):
                        self.param_widgets[param].set(value)
                    elif isinstance(self.param_widgets[param], tk.BooleanVar):
                        self.param_widgets[param].set(value)
                    elif isinstance(self.param_widgets[param], ttk.Combobox):
                        self.param_widgets[param].set(value)
            
            return True
        return False
    
    def clear_parameter_storage(self):
        """Clear all stored parameters and results"""
        if self.file_parameter_storage or self.file_results_storage:
            result = messagebox.askyesno("Clear Parameter Storage", 
                                       f"Clear stored parameters for {len(self.file_parameter_storage)} files?")
            if result:
                self.file_parameter_storage.clear()
                self.file_results_storage.clear()
                self.update_parameter_storage_info()
                messagebox.showinfo("Cleared", "Parameter storage cleared.")
        else:
            messagebox.showinfo("Empty", "No stored parameters to clear.")
    
    def export_parameter_database(self):
        """Export all stored parameters and results to CSV"""
        if not self.file_parameter_storage:
            messagebox.showwarning("No Data", "No stored parameters to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Parameter Database",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if filename:
            try:
                export_rows = []
                
                for audio_filename, params in self.file_parameter_storage.items():
                    results = self.file_results_storage.get(audio_filename, {})
                    
                    row = {
                        'audio_file': audio_filename,
                        'detected_events': len(results.get('events', [])),
                        'threshold_used': results.get('threshold', 0),
                        'timestamp': results.get('timestamp', ''),
                    }
                    
                    # Add all parameters
                    for param, value in params.items():
                        row[f'param_{param}'] = value
                    
                    # Add ground truth info if available
                    ground_truth = results.get('ground_truth')
                    if ground_truth:
                        row.update({
                            'expected_swallows': ground_truth.swallow_count,
                            'group': ground_truth.group,
                            'volume_ml': ground_truth.volume_ml,
                            'accuracy_error': abs(len(results.get('events', [])) - ground_truth.swallow_count)
                        })
                    
                    export_rows.append(row)
                
                df = pd.DataFrame(export_rows)
                df.to_csv(filename, index=False)
                
                messagebox.showinfo("Success", 
                                  f"Parameter database exported to {filename}\n\n"
                                  f"Exported data for {len(export_rows)} files")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export parameter database:\n{str(e)}")
    
    def update_parameter_storage_info(self):
        """Update parameter storage information display"""
        count = len(self.file_parameter_storage)
        if count == 0:
            self.param_storage_label.config(text="No parameters stored", foreground='blue')
        else:
            # Show which file is currently stored
            current_file = self.current_data['filename'] if self.current_data else None
            is_current_stored = current_file in self.file_parameter_storage if current_file else False
            
            status_text = f"{count} file(s) with saved parameters"
            if is_current_stored:
                status_text += f"\nCurrent file: SAVED ✓"
            else:
                status_text += f"\nCurrent file: NOT SAVED"
            
            self.param_storage_label.config(text=status_text, foreground='green' if count > 0 else 'blue')
    
    def add_current_spl_to_storage(self):
        """Add current file's SPL analysis to storage for batch export"""
        if not self.current_data or not self.current_data['events']:
            messagebox.showwarning("No Data", "No swallow events detected in current file!")
            return
        
        events = self.current_data['events']
        y = self.current_data['y_clean']
        sr = self.current_data['sr']
        
        # Get pre and post segments
        first_swallow_start = events[0]['start']
        last_swallow_end = events[-1]['end']
        
        pre_start_time = max(0, first_swallow_start - 0.5)
        pre_end_time = first_swallow_start
        
        post_start_time = last_swallow_end
        post_end_time = min(len(y)/sr, last_swallow_end + 0.5)
        
        # Extract segments
        pre_start_idx = int(pre_start_time * sr)
        pre_end_idx = int(pre_end_time * sr)
        post_start_idx = int(post_start_time * sr)
        post_end_idx = int(post_end_time * sr)
        
        pre_segment = y[pre_start_idx:pre_end_idx]
        post_segment = y[post_start_idx:post_end_idx]
        
        # Analyze segments and store results
        spl_data_entry = {
            'filename': self.current_data['filename'],
            'detected_swallows': len(events),
            'pre_frequencies': None,
            'pre_spl_values': None,
            'post_frequencies': None,
            'post_spl_values': None
        }
        
        # Add ground truth info if available
        if self.current_ground_truth:
            spl_data_entry.update({
                'expected_swallows': self.current_ground_truth.swallow_count,
                'group': self.current_ground_truth.group,
                'volume_ml': self.current_ground_truth.volume_ml
            })
        
        if len(pre_segment) >= 2048:
            pre_freqs, pre_spl = self.calculate_spl_vs_frequency_detailed(pre_segment, sr)
            if pre_freqs is not None:
                spl_data_entry['pre_frequencies'] = pre_freqs
                spl_data_entry['pre_spl_values'] = pre_spl
        
        if len(post_segment) >= 2048:
            post_freqs, post_spl = self.calculate_spl_vs_frequency_detailed(post_segment, sr)
            if post_freqs is not None:
                spl_data_entry['post_frequencies'] = post_freqs
                spl_data_entry['post_spl_values'] = post_spl
        
        # Add to storage
        self.spl_analysis_storage.append(spl_data_entry)
        
        # Update SPL storage display automatically
        self.update_spl_storage_info()
        
        messagebox.showinfo("Added to Storage", 
                          f"SPL data for '{self.current_data['filename']}' added to batch storage.\n"
                          f"Total files in storage: {len(self.spl_analysis_storage)}")
    
    def export_batch_spl_data(self):
        """Export all stored SPL data to a single CSV file"""
        if not self.spl_analysis_storage:
            messagebox.showwarning("No Data", "No SPL analysis data in storage!\n\nUse 'Add Current SPL to Storage' first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Batch SPL Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if filename:
            try:
                # Create comprehensive export data
                export_rows = []
                
                for entry in self.spl_analysis_storage:
                    # Process pre-swallow data
                    if entry['pre_frequencies'] is not None and entry['pre_spl_values'] is not None:
                        for freq, spl in zip(entry['pre_frequencies'], entry['pre_spl_values']):
                            row = {
                                'audio_file': entry['filename'],
                                'segment_type': 'pre_swallow',
                                'frequency_hz': freq,
                                'spl_db': spl,
                                'detected_swallows': entry['detected_swallows']
                            }
                            
                            # Add ground truth info if available
                            if 'expected_swallows' in entry:
                                row.update({
                                    'expected_swallows': entry['expected_swallows'],
                                    'group': entry['group'],
                                    'volume_ml': entry['volume_ml']
                                })
                            
                            export_rows.append(row)
                    
                    # Process post-swallow data
                    if entry['post_frequencies'] is not None and entry['post_spl_values'] is not None:
                        for freq, spl in zip(entry['post_frequencies'], entry['post_spl_values']):
                            row = {
                                'audio_file': entry['filename'],
                                'segment_type': 'post_swallow',
                                'frequency_hz': freq,
                                'spl_db': spl,
                                'detected_swallows': entry['detected_swallows']
                            }
                            
                            # Add ground truth info if available
                            if 'expected_swallows' in entry:
                                row.update({
                                    'expected_swallows': entry['expected_swallows'],
                                    'group': entry['group'],
                                    'volume_ml': entry['volume_ml']
                                })
                            
                            export_rows.append(row)
                
                df = pd.DataFrame(export_rows)
                df.to_csv(filename, index=False)
                
                messagebox.showinfo("Success", 
                                  f"Batch SPL data exported to {filename}\n\n"
                                  f"Exported data from {len(self.spl_analysis_storage)} files\n"
                                  f"Total frequency measurements: {len(export_rows)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export batch SPL data:\n{str(e)}")
    
    def clear_spl_storage(self):
        """Clear the SPL analysis storage"""
        if self.spl_analysis_storage:
            result = messagebox.askyesno("Clear Storage", 
                                       f"Clear {len(self.spl_analysis_storage)} stored SPL analyses?")
            if result:
                self.spl_analysis_storage = []
                self.update_spl_storage_info()
                messagebox.showinfo("Cleared", "SPL analysis storage cleared.")
        else:
            messagebox.showinfo("Empty", "SPL analysis storage is already empty.")
    
    def show_spl_frequency_analysis(self):
        """Show enhanced SPL vs Frequency analysis window with light styling"""
        if not self.current_data or not self.current_data['events']:
            messagebox.showwarning("No Data", "No swallow events detected in current file!")
            return
        
        events = self.current_data['events']
        y = self.current_data['y_clean']
        sr = self.current_data['sr']
        
        # Get pre and post segments
        first_swallow_start = events[0]['start']
        last_swallow_end = events[-1]['end']
        
        pre_start_time = max(0, first_swallow_start - 0.5)
        pre_end_time = first_swallow_start
        
        post_start_time = last_swallow_end
        post_end_time = min(len(y)/sr, last_swallow_end + 0.5)
        
        # Extract segments
        pre_start_idx = int(pre_start_time * sr)
        pre_end_idx = int(pre_end_time * sr)
        post_start_idx = int(post_start_time * sr)
        post_end_idx = int(post_end_time * sr)
        
        pre_segment = y[pre_start_idx:pre_end_idx]
        post_segment = y[post_start_idx:post_end_idx]
        
        if len(pre_segment) < 2048 and len(post_segment) < 2048:
            messagebox.showwarning("Insufficient Data", "Not enough audio data for detailed SPL analysis!")
            return
        
        # Create analysis window
        spl_window = tk.Toplevel(self.root)
        spl_window.title("Enhanced SPL vs Frequency Analysis")
        spl_window.geometry("1000x700")
        
        # Create matplotlib figure with light theme
        fig = Figure(figsize=(10, 7), dpi=100)
        canvas = FigureCanvasTkAgg(fig, spl_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Analyze segments with higher resolution
        spl_analysis_data = {}
        
        if len(pre_segment) >= 2048:
            pre_freqs, pre_spl = self.calculate_spl_vs_frequency_detailed(pre_segment, sr)
            if pre_freqs is not None:
                spl_analysis_data['pre_swallow'] = {
                    'frequencies': pre_freqs,
                    'spl_values': pre_spl,
                    'segment_duration': len(pre_segment) / sr
                }
        
        if len(post_segment) >= 2048:
            post_freqs, post_spl = self.calculate_spl_vs_frequency_detailed(post_segment, sr)
            if post_freqs is not None:
                spl_analysis_data['post_swallow'] = {
                    'frequencies': post_freqs,
                    'spl_values': post_spl,
                    'segment_duration': len(post_segment) / sr
                }
        
        # Plot results with light theme
        fig.clear()
        
        if spl_analysis_data:
            ax = fig.add_subplot(111)
            
            # Define colors and styles
            colors = {'pre_swallow': '#1f77b4', 'post_swallow': '#ff7f0e'}  # Blue and Orange
            labels = {'pre_swallow': 'Pre', 'post_swallow': 'Post'}
            
            for segment_type, data in spl_analysis_data.items():
                frequencies = data['frequencies']
                spl_values = data['spl_values']
                
                # Plot with detailed appearance
                ax.plot(frequencies, spl_values, 
                       linewidth=1, 
                       color=colors[segment_type], 
                       alpha=0.8,
                       label=labels[segment_type])
            
            # Formatting the plot
            ax.set_xlabel('Frequency (Hz)', fontsize=14)
            ax.set_ylabel('SPL (dB/Hz)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12, loc='upper right')
            
            # Set frequency range (0-2000 Hz)
            ax.set_xlim(0, 2000)
            
            # Set reasonable SPL limits
            all_spl_values = []
            for data in spl_analysis_data.values():
                all_spl_values.extend(data['spl_values'])
            
            if all_spl_values:
                all_spl_values = np.array(all_spl_values)
                valid_spl = all_spl_values[np.isfinite(all_spl_values)]
                if len(valid_spl) > 0:
                    y_min = np.percentile(valid_spl, 5) - 5
                    y_max = np.percentile(valid_spl, 95) + 5
                    ax.set_ylim(y_min, y_max)
            
            # Create title in the style: "FILENAME - Pre first - Post last"
            filename_base = self.current_data["filename"].replace('.wav', '').replace('.WAV', '')
            ax.set_title(f'{filename_base} - Pre first - Post last', 
                        fontsize=16, fontweight='bold', pad=20)
            
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No valid SPL data available\n\nSegments too short for detailed analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14, weight='bold')
            ax.axis('off')
        
        fig.tight_layout()
        canvas.draw()
        
        # Add control buttons
        button_frame = ttk.Frame(spl_window)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Add to Storage for Batch Export", 
                  command=self.add_current_spl_to_storage).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Batch SPL Data", 
                  command=self.export_batch_spl_data).pack(side=tk.RIGHT, padx=5)
    
    def create_gui(self):
        """Create the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("Interactive Swallow Detection Parameter Tuner")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Create main frames
        self.create_menu()
        self.create_control_panel()
        self.create_plot_area()
        self.create_status_bar()
        
        # Initialize with empty plot
        self.setup_empty_plot()
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Audio Folder", command=self.open_audio_folder)
        file_menu.add_command(label="Load Excel Ground Truth", command=self.load_excel_ground_truth)
        file_menu.add_separator()
        file_menu.add_command(label="Save Parameters", command=self.save_parameters)
        file_menu.add_command(label="Load Parameters", command=self.load_parameters)
        file_menu.add_command(label="Load Parameters from CSV", command=self.load_parameters_from_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="SPL vs Frequency Analysis", command=self.show_spl_frequency_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Save Current Analysis", command=self.save_current_analysis_state)
        analysis_menu.add_command(label="Export Parameter Database", command=self.export_parameter_database)
        analysis_menu.add_command(label="Clear Parameter Storage", command=self.clear_parameter_storage)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Add Current SPL to Storage", command=self.add_current_spl_to_storage)
        analysis_menu.add_command(label="Export Batch SPL Data", command=self.export_batch_spl_data)
        analysis_menu.add_command(label="Clear SPL Storage", command=self.clear_spl_storage)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Accuracy Report", command=self.show_accuracy_report)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Reset Parameters", command=self.reset_parameters)
        tools_menu.add_command(label="Auto-Tune", command=self.auto_tune_parameters)
    
    def create_control_panel(self):
        """Create the scrollable parameter control panel"""
        # Create main control frame
        control_main_frame = ttk.Frame(self.root)
        control_main_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Create canvas and scrollbar for scrolling
        canvas = tk.Canvas(control_main_frame, width=350, height=600)
        scrollbar = ttk.Scrollbar(control_main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrolling components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mouse wheel binding for scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # File navigation
        nav_frame = ttk.LabelFrame(scrollable_frame, text="Audio File Navigation", padding=10)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(nav_frame, text="◀ Previous", command=self.prev_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_file).pack(side=tk.LEFT, padx=5)
        
        self.info_label = ttk.Label(nav_frame, text="No files loaded")
        self.info_label.pack(side=tk.LEFT, padx=20)
        
        # Ground truth info
        gt_frame = ttk.LabelFrame(scrollable_frame, text="Ground Truth Information", padding=5)
        gt_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.ground_truth_label = ttk.Label(gt_frame, text="No ground truth loaded", 
                                          foreground='red', wraplength=300)
        self.ground_truth_label.pack(pady=5)
        
        # Parameter Storage info
        param_frame = ttk.LabelFrame(scrollable_frame, text="Parameter Storage", padding=5)
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.param_storage_label = ttk.Label(param_frame, text="No parameters stored", 
                                           foreground='blue', wraplength=300)
        self.param_storage_label.pack(pady=2)
        
        ttk.Button(param_frame, text="Save Current Analysis", 
                  command=self.save_current_analysis_state).pack(fill=tk.X, pady=1)
        
        # SPL Storage info
        spl_frame = ttk.LabelFrame(scrollable_frame, text="SPL Analysis Storage", padding=5)
        spl_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.spl_storage_label = ttk.Label(spl_frame, text="No SPL data stored", 
                                         foreground='blue', wraplength=300)
        self.spl_storage_label.pack(pady=2)
        
        # Parameter controls (in scrollable frame)
        params_frame = ttk.LabelFrame(scrollable_frame, text="Detection Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create parameter widgets
        self.create_parameter_widgets(params_frame)
        
        # Action buttons
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="Update Analysis", 
                  command=self.update_analysis, style='Accent.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Save Current Analysis", 
                  command=self.save_current_analysis_state).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="SPL Analysis", 
                  command=self.show_spl_frequency_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Add SPL to Storage", 
                  command=self.add_current_spl_to_storage).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Export Batch SPL", 
                  command=self.export_batch_spl_data).pack(fill=tk.X, pady=2)
    
    def update_spl_storage_info(self):
        """Update SPL storage information display"""
        count = len(self.spl_analysis_storage)
        if count == 0:
            self.spl_storage_label.config(text="No SPL data stored", foreground='blue')
        else:
            self.spl_storage_label.config(text=f"{count} file(s) in SPL storage\nReady for batch export", 
                                        foreground='green')
    
    def create_parameter_widgets(self, parent):
        """Create widgets for parameter adjustment"""
        
        # Threshold Parameters
        thresh_frame = ttk.LabelFrame(parent, text="🎯 Threshold Sensitivity", padding=5)
        thresh_frame.pack(fill=tk.X, pady=5)
        
        # Energy Percentile Slider
        ttk.Label(thresh_frame, text="Energy Percentile:").pack(anchor=tk.W)
        self.param_widgets['energy_percentile'] = tk.Scale(
            thresh_frame, from_=20, to=90, orient=tk.HORIZONTAL,
            command=lambda x: self.on_parameter_change('energy_percentile', float(x))
        )
        self.param_widgets['energy_percentile'].set(self.detection_params['energy_percentile'])
        self.param_widgets['energy_percentile'].pack(fill=tk.X)
        
        # Adaptive Threshold Checkbox
        self.param_widgets['adaptive_threshold'] = tk.BooleanVar(value=self.detection_params['adaptive_threshold'])
        ttk.Checkbutton(thresh_frame, text="Use Adaptive Threshold", 
                       variable=self.param_widgets['adaptive_threshold'],
                       command=lambda: self.on_parameter_change('adaptive_threshold', 
                                                               self.param_widgets['adaptive_threshold'].get())).pack(anchor=tk.W)
        
        # Boundary Parameters
        boundary_frame = ttk.LabelFrame(parent, text="🔍 Event Boundaries", padding=5)
        boundary_frame.pack(fill=tk.X, pady=5)
        
        # Boundary Method
        ttk.Label(boundary_frame, text="Boundary Method:").pack(anchor=tk.W)
        self.param_widgets['boundary_method'] = ttk.Combobox(
            boundary_frame, values=['energy_based', 'silence_based'], state='readonly')
        self.param_widgets['boundary_method'].set(self.detection_params['boundary_method'])
        self.param_widgets['boundary_method'].bind('<<ComboboxSelected>>', 
                                                  lambda e: self.on_parameter_change('boundary_method',
                                                                                   self.param_widgets['boundary_method'].get()))
        self.param_widgets['boundary_method'].pack(fill=tk.X)
        
        # Energy Drop Threshold
        ttk.Label(boundary_frame, text="Energy Drop Threshold:").pack(anchor=tk.W)
        self.param_widgets['energy_drop_threshold'] = tk.Scale(
            boundary_frame, from_=0.1, to=0.8, resolution=0.1, orient=tk.HORIZONTAL,
            command=lambda x: self.on_parameter_change('energy_drop_threshold', float(x))
        )
        self.param_widgets['energy_drop_threshold'].set(self.detection_params['energy_drop_threshold'])
        self.param_widgets['energy_drop_threshold'].pack(fill=tk.X)
        
        # Event Merging Parameters
        merging_frame = ttk.LabelFrame(parent, text="🔗 Event Merging", padding=5)
        merging_frame.pack(fill=tk.X, pady=5)
        
        # Merge Close Events
        self.param_widgets['merge_close_events'] = tk.BooleanVar(value=self.detection_params['merge_close_events'])
        ttk.Checkbutton(merging_frame, text="Merge Close Events", 
                       variable=self.param_widgets['merge_close_events'],
                       command=lambda: self.on_parameter_change('merge_close_events', 
                                                               self.param_widgets['merge_close_events'].get())).pack(anchor=tk.W)
        
        # Min Event Separation
        ttk.Label(merging_frame, text="Min Event Separation (s):").pack(anchor=tk.W)
        self.param_widgets['min_event_separation'] = tk.Scale(
            merging_frame, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
            command=lambda x: self.on_parameter_change('min_event_separation', float(x))
        )
        self.param_widgets['min_event_separation'].set(self.detection_params['min_event_separation'])
        self.param_widgets['min_event_separation'].pack(fill=tk.X)
        
        # Duration Parameters
        duration_frame = ttk.LabelFrame(parent, text="⏱️ Duration Limits", padding=5)
        duration_frame.pack(fill=tk.X, pady=5)
        
        # Min Duration
        ttk.Label(duration_frame, text="Min Duration (s):").pack(anchor=tk.W)
        self.param_widgets['min_swallow_duration'] = tk.Scale(
            duration_frame, from_=0.05, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
            command=lambda x: self.on_parameter_change('min_swallow_duration', float(x))
        )
        self.param_widgets['min_swallow_duration'].set(self.detection_params['min_swallow_duration'])
        self.param_widgets['min_swallow_duration'].pack(fill=tk.X)
        
        # Max Duration
        ttk.Label(duration_frame, text="Max Duration (s):").pack(anchor=tk.W)
        self.param_widgets['max_swallow_duration'] = tk.Scale(
            duration_frame, from_=1.0, to=5.0, resolution=0.5, orient=tk.HORIZONTAL,
            command=lambda x: self.on_parameter_change('max_swallow_duration', float(x))
        )
        self.param_widgets['max_swallow_duration'].set(self.detection_params['max_swallow_duration'])
        self.param_widgets['max_swallow_duration'].pack(fill=tk.X)
        
        # Fine-tuning Parameters
        fine_frame = ttk.LabelFrame(parent, text="🔧 Fine-tuning", padding=5)
        fine_frame.pack(fill=tk.X, pady=5)
        
        # Smoothing Sigma
        ttk.Label(fine_frame, text="Smoothing:").pack(anchor=tk.W)
        self.param_widgets['smoothing_sigma'] = tk.Scale(
            fine_frame, from_=1, to=5, orient=tk.HORIZONTAL,
            command=lambda x: self.on_parameter_change('smoothing_sigma', int(x))
        )
        self.param_widgets['smoothing_sigma'].set(self.detection_params['smoothing_sigma'])
        self.param_widgets['smoothing_sigma'].pack(fill=tk.X)
        
        # Min Prominence
        ttk.Label(fine_frame, text="Min Prominence:").pack(anchor=tk.W)
        self.param_widgets['min_prominence'] = tk.Scale(
            fine_frame, from_=0.001, to=0.1, resolution=0.001, orient=tk.HORIZONTAL,
            command=lambda x: self.on_parameter_change('min_prominence', float(x))
        )
        self.param_widgets['min_prominence'].set(self.detection_params['min_prominence'])
        self.param_widgets['min_prominence'].pack(fill=tk.X)
    
    def create_plot_area(self):
        """Create the matplotlib plot area"""
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        
        # Navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        
        # Pack canvas
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Label(self.root, text="Ready - Load audio files to begin", 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_empty_plot(self):
        """Setup empty plot when no data is loaded"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Load Audio Files to Begin\n(File → Open Audio Folder)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()
    
    def open_audio_folder(self):
        """Open audio folder dialog"""
        folder = filedialog.askdirectory(title="Select Audio Folder")
        if folder:
            self.audio_folder = Path(folder)
            self.load_audio_files()
    
    def load_audio_files(self):
        """Load audio files from selected folder"""
        if not self.audio_folder:
            return
        
        # Load audio files and remove duplicates
        wav_files = list(self.audio_folder.glob("*.wav"))
        WAV_files = list(self.audio_folder.glob("*.WAV"))
        
        # Combine and remove duplicates by converting to set of resolved paths
        all_files = wav_files + WAV_files
        unique_files = []
        seen_paths = set()
        
        for file_path in all_files:
            resolved_path = file_path.resolve()
            if resolved_path not in seen_paths:
                seen_paths.add(resolved_path)
                unique_files.append(file_path)
        
        self.audio_files = sorted(unique_files)
        
        if not self.audio_files:
            messagebox.showwarning("No Files", "No audio files found in selected folder!")
            return
        
        self.current_file_idx = 0
        self.status_bar.config(text=f"Loaded {len(self.audio_files)} audio files")
        self.update_file_info()
        self.load_current_file()
    
    def load_excel_ground_truth(self):
        """Load Excel ground truth file"""
        filename = filedialog.askopenfilename(
            title="Select Excel Ground Truth File",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if filename:
            success, message = self.ground_truth_manager.load_excel_file(filename)
            if success:
                messagebox.showinfo("Success", message)
                self.update_ground_truth_info()
            else:
                messagebox.showerror("Error", message)
    
    def update_ground_truth_info(self):
        """Update ground truth information display"""
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            current_file = self.audio_files[self.current_file_idx]
            self.current_ground_truth = self.ground_truth_manager.get_ground_truth(current_file.name)
            
            if self.current_ground_truth:
                info = (f"Ground Truth Found:\n"
                       f"Expected Swallows: {self.current_ground_truth.swallow_count}\n"
                       f"Group: {self.current_ground_truth.group}")
                if self.current_ground_truth.volume_ml:
                    info += f"\nVolume: {self.current_ground_truth.volume_ml}mL"
                self.ground_truth_label.config(text=info, foreground='green')
            else:
                self.ground_truth_label.config(text="No matching ground truth found", foreground='orange')
        else:
            self.ground_truth_label.config(text="No ground truth loaded", foreground='red')
    
    def load_current_file(self):
        """Load and analyze current audio file"""
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            return
        
        audio_file = self.audio_files[self.current_file_idx]
        self.status_bar.config(text=f"Loading {audio_file.name}...")
        
        # Update ground truth for current file
        self.update_ground_truth_info()
        
        # Check for saved parameters and load them
        if self.load_analysis_state_for_file(audio_file.name):
            # Parameters loaded from storage
            pass
        
        try:
            def load_audio():
                y, sr = librosa.load(audio_file, sr=self.sr)
                self.root.after(0, lambda: self.process_audio_data(y, sr, audio_file.name))
            
            thread = threading.Thread(target=load_audio)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {audio_file.name}:\n{str(e)}")
            self.status_bar.config(text="Error loading file")
    
    def process_audio_data(self, y, sr, filename):
        """Process loaded audio data"""
        try:
            events, y_clean, rms_smooth, threshold = self.detect_swallow_events(y, sr)
            
            time_audio = np.linspace(0, len(y) / sr, len(y))
            time_rms = np.linspace(0, len(y) / sr, len(rms_smooth))
            
            self.current_data = {
                'filename': filename,
                'y_original': y,
                'y_clean': y_clean,
                'rms_smooth': rms_smooth,
                'events': events,
                'time_audio': time_audio,
                'time_rms': time_rms,
                'threshold': threshold,
                'sr': sr
            }
            
            self.update_plot()
            
            # Update parameter storage display
            self.update_parameter_storage_info()
            
            status_text = f"Analyzed {filename} - {len(events)} events detected"
            if self.current_ground_truth:
                expected = self.current_ground_truth.swallow_count
                error = abs(len(events) - expected)
                accuracy = "✓" if error == 0 else "✗"
                status_text += f" (Expected: {expected}, Error: {error} {accuracy})"
            
            self.status_bar.config(text=status_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze audio:\n{str(e)}")
            self.status_bar.config(text="Error analyzing audio")
    
    def update_plot(self):
        """Update the plot with current data"""
        if not self.current_data:
            return
        
        data = self.current_data
        
        self.fig.clear()
        
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1])
        
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax2 = self.fig.add_subplot(gs[1, 0])
        ax_stats = self.fig.add_subplot(gs[:, 1])
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Plot 1: RMS Energy
        ax1.plot(data['time_rms'], data['rms_smooth'], color='green', linewidth=2, label='RMS Energy')
        ax1.axhline(y=data['threshold'], color='orange', linestyle='-', linewidth=2, 
                   label=f'Threshold ({data["threshold"]:.4f})')
        
        for i, event in enumerate(data['events']):
            color = colors[i % len(colors)]
            ax1.axvspan(event['start'], event['end'], alpha=0.2, color=color)
            ax1.plot(event['peak_time'], event['peak_energy'], 'o', color=color, markersize=8)
        
        ax1.set_title('RMS Energy Analysis')
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cleaned waveform with labels
        ax2.plot(data['time_audio'], data['y_clean'], alpha=0.8, color='navy', linewidth=0.8)
        for i, event in enumerate(data['events']):
            color = colors[i % len(colors)]
            ax2.axvspan(event['start'], event['end'], alpha=0.3, color=color)
            mid_time = (event['start'] + event['end']) / 2
            ax2.text(mid_time, ax2.get_ylim()[1]*0.8, f'S{i+1}', 
                    ha='center', va='center', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        ax2.set_title('Processed Audio with Event Labels')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        # Enhanced statistics panel with ground truth comparison
        stats_text = f"""CURRENT FILE:
{data['filename']}

DETECTION RESULTS:
Events Detected: {len(data['events'])}
Threshold: {data['threshold']:.4f}
File Duration: {len(data['y_original'])/data['sr']:.1f}s

GROUND TRUTH:"""
        
        if self.current_ground_truth:
            expected = self.current_ground_truth.swallow_count
            detected = len(data['events'])
            error = abs(detected - expected)
            accuracy = "✓" if error == 0 else "✗"
            
            stats_text += f"""
Expected: {expected}
Detected: {detected}
Error: {error} {accuracy}
Group: {self.current_ground_truth.group}"""
            if self.current_ground_truth.volume_ml:
                stats_text += f"""
Volume: {self.current_ground_truth.volume_ml}mL"""
        else:
            stats_text += "\nNo ground truth available"
        
        stats_text += f"""

SPL STORAGE:
Files stored: {len(self.spl_analysis_storage)}

PARAMETER STORAGE:
Files with saved parameters: {len(self.file_parameter_storage)}
Current file saved: {"✓" if (self.current_data and self.current_data['filename'] in self.file_parameter_storage) else "✗"}

PARAMETERS:
Energy %tile: {self.detection_params['energy_percentile']}
Adaptive: {self.detection_params['adaptive_threshold']}
Boundary: {self.detection_params['boundary_method']}
Energy Drop: {self.detection_params['energy_drop_threshold']}
Merge Events: {self.detection_params['merge_close_events']}
Min Separation: {self.detection_params['min_event_separation']}s

EVENT DETAILS:"""
        
        if data['events']:
            for i, event in enumerate(data['events'][:5]):
                stats_text += f"\n{i+1}: {event['start']:.2f}-{event['end']:.2f}s"
                stats_text += f" (dur: {event['duration']:.2f}s)"
        else:
            stats_text += "\nNo events detected"
        
        # Add Excel event durations if available
        if self.current_ground_truth and self.current_ground_truth.event_durations:
            stats_text += f"""

EXCEL EVENT DURATIONS:"""
            for i, duration in enumerate(self.current_ground_truth.event_durations[:8]):
                stats_text += f"\nEvent {i+1}: {duration:.2f}s"
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                     fontsize=9, verticalalignment='top', fontfamily='monospace')
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis('off')
        
        self.fig.suptitle(f'Swallow Detection Analysis - {data["filename"]}', fontsize=14, fontweight='bold')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def on_parameter_change(self, param_name, value):
        """Handle parameter change from GUI widgets"""
        self.detection_params[param_name] = value
        
        if self.current_data:
            if hasattr(self, '_update_timer'):
                self.root.after_cancel(self._update_timer)
            self._update_timer = self.root.after(200, self.update_analysis)
    
    def update_analysis(self):
        """Update analysis with current parameters"""
        if self.current_data:
            y = self.current_data['y_original']
            sr = self.current_data['sr']
            filename = self.current_data['filename']
            self.process_audio_data(y, sr, filename)
    
    def prev_file(self):
        """Navigate to previous file with auto-save/load"""
        if self.audio_files and self.current_file_idx > 0:
            # Auto-save current analysis if data exists
            if self.current_data:
                self.save_current_analysis_state()
            
            self.current_file_idx -= 1
            self.update_file_info()
            self.load_current_file()
    
    def next_file(self):
        """Navigate to next file with auto-save/load"""
        if self.audio_files and self.current_file_idx < len(self.audio_files) - 1:
            # Auto-save current analysis if data exists
            if self.current_data:
                self.save_current_analysis_state()
            
            self.current_file_idx += 1
            self.update_file_info()
            self.load_current_file()
    
    def update_file_info(self):
        """Update file navigation info"""
        if self.audio_files:
            self.info_label.config(text=f"File {self.current_file_idx + 1} of {len(self.audio_files)}")
        else:
            self.info_label.config(text="No files loaded")
    
    def show_accuracy_report(self):
        """Show accuracy report if ground truth is available"""
        if not self.ground_truth_manager.ground_truth_data:
            messagebox.showwarning("No Ground Truth", "Load Excel ground truth file first!")
            return
        
        report_window = tk.Toplevel(self.root)
        report_window.title("Accuracy Report")
        report_window.geometry("600x500")
        
        text_widget = tk.Text(report_window, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(report_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        report = "SWALLOW DETECTION ACCURACY REPORT\n"
        report += "=" * 50 + "\n\n"
        
        total_files = len(self.ground_truth_manager.ground_truth_data)
        report += f"Total ground truth entries: {total_files}\n"
        report += f"Total audio files loaded: {len(self.audio_files)}\n"
        report += f"SPL analyses stored: {len(self.spl_analysis_storage)}\n"
        report += f"Parameter sets stored: {len(self.file_parameter_storage)}\n\n"
        
        if self.current_ground_truth and self.current_data:
            detected = len(self.current_data['events'])
            expected = self.current_ground_truth.swallow_count
            error = abs(detected - expected)
            
            report += "CURRENT FILE ANALYSIS:\n"
            report += f"File: {self.current_data['filename']}\n"
            report += f"Expected swallows: {expected}\n"
            report += f"Detected swallows: {detected}\n"
            report += f"Error: {error}\n"
            report += f"Accuracy: {'Perfect' if error == 0 else f'Off by {error}'}\n"
            report += f"Group: {self.current_ground_truth.group}\n"
            if self.current_ground_truth.volume_ml:
                report += f"Volume: {self.current_ground_truth.volume_ml}mL\n"
        else:
            report += "CURRENT FILE: No ground truth match or no file loaded\n"
        
        report += "\nTo generate a full accuracy report, process multiple files\n"
        report += "and use the batch processing features.\n"
        
        text_widget.insert(1.0, report)
        text_widget.config(state=tk.DISABLED)
        
        text_widget.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
    
    def reset_parameters(self):
        """Reset parameters to defaults"""
        default_params = {
            'min_swallow_duration': 0.1,
            'max_swallow_duration': 3.0,
            'min_event_separation': 0.5,
            'energy_percentile': 60,
            'adaptive_threshold': True,
            'min_prominence': 0.01,
            'smoothing_sigma': 2,
            'boundary_method': 'energy_based',
            'energy_drop_threshold': 0.3,
            'merge_close_events': True,
        }
        
        self.detection_params.update(default_params)
        
        for param, value in default_params.items():
            if param in self.param_widgets:
                if isinstance(self.param_widgets[param], tk.Scale):
                    self.param_widgets[param].set(value)
                elif isinstance(self.param_widgets[param], tk.BooleanVar):
                    self.param_widgets[param].set(value)
                elif isinstance(self.param_widgets[param], ttk.Combobox):
                    self.param_widgets[param].set(value)
        
        self.update_analysis()
        messagebox.showinfo("Reset", "Parameters reset to defaults")
    
    def save_parameters(self):
        """Save current parameters to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.detection_params, f, indent=2)
                messagebox.showinfo("Saved", f"Parameters saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save parameters:\n{str(e)}")
    
    def load_parameters(self):
        """Load parameters from file"""
        filename = filedialog.askopenfilename(
            title="Load Parameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    params = json.load(f)
                
                self.detection_params.update(params)
                
                for param, value in params.items():
                    if param in self.param_widgets:
                        if isinstance(self.param_widgets[param], tk.Scale):
                            self.param_widgets[param].set(value)
                        elif isinstance(self.param_widgets[param], tk.BooleanVar):
                            self.param_widgets[param].set(value)
                        elif isinstance(self.param_widgets[param], ttk.Combobox):
                            self.param_widgets[param].set(value)
                
                self.update_analysis()
                messagebox.showinfo("Loaded", f"Parameters loaded from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load parameters:\n{str(e)}")

    def load_parameters_from_csv_bulk(self):
        """Load parameters for ALL files from exported CSV parameter database"""
        filename = filedialog.askopenfilename(
            title="Load Parameters from CSV (All Files)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Read the CSV file
            df = pd.read_csv(filename)
            
            if df.empty:
                messagebox.showwarning("Empty File", "The CSV file is empty!")
                return
            
            # Get parameter columns (those starting with 'param_')
            param_columns = [col for col in df.columns if col.startswith('param_')]
            
            if not param_columns:
                messagebox.showwarning("No Parameters", "No parameter columns found in CSV file!")
                return
            
            # Load parameters for all files in the CSV
            loaded_files = 0
            skipped_files = 0
            
            for idx, row in df.iterrows():
                audio_file = row.get('audio_file', '')
                if not audio_file:
                    skipped_files += 1
                    continue
                
                # Extract parameters for this file
                file_params = {}
                
                for col in param_columns:
                    # Remove 'param_' prefix to get actual parameter name
                    param_name = col.replace('param_', '')
                    param_value = row[col]
                    
                    # Handle different parameter types
                    if param_name in self.detection_params:
                        # Convert to appropriate type based on current parameter
                        current_value = self.detection_params[param_name]
                        if isinstance(current_value, bool):
                            file_params[param_name] = bool(param_value)
                        elif isinstance(current_value, int):
                            file_params[param_name] = int(param_value)
                        elif isinstance(current_value, float):
                            file_params[param_name] = float(param_value)
                        else:
                            file_params[param_name] = param_value
                
                # Store parameters for this file
                if file_params:
                    self.file_parameter_storage[audio_file] = file_params
                    loaded_files += 1
                else:
                    skipped_files += 1
            
            # Update parameter storage display
            self.update_parameter_storage_info()
            
            # If current file has loaded parameters, apply them
            if self.current_data and self.current_data['filename'] in self.file_parameter_storage:
                self.load_analysis_state_for_file(self.current_data['filename'])
                self.update_analysis()
            
            # Show success message
            messagebox.showinfo("Bulk Parameters Loaded", 
                              f"Parameters loaded from CSV\n"
                              f"Successfully loaded: {loaded_files} files\n"
                              f"Skipped: {skipped_files} files\n"
                              f"Total in parameter storage: {len(self.file_parameter_storage)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load parameters from CSV:\n{str(e)}")

    def load_parameters_from_csv(self):
        """Load parameters from exported CSV parameter database - enhanced with bulk option"""
        # Create a dialog to choose single file or bulk load
        choice_window = tk.Toplevel(self.root)
        choice_window.title("CSV Parameter Loading Options")
        choice_window.geometry("400x200")
        choice_window.transient(self.root)
        choice_window.grab_set()
        
        ttk.Label(choice_window, text="Choose how to load CSV parameters:", 
                  font=('Arial', 12)).pack(pady=20)
        
        choice = ['single']  # Use list for mutable reference
        
        def load_single():
            choice[0] = 'single'
            choice_window.destroy()
        
        def load_bulk():
            choice[0] = 'bulk'
            choice_window.destroy()
        
        button_frame = ttk.Frame(choice_window)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Load Single File Parameters\n(Apply to current analysis)", 
                   command=load_single).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Load All Files Parameters\n(Store for each file)", 
                   command=load_bulk).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(choice_window, text="Single: Loads one set and applies now\n"
                                      "Bulk: Loads all files for auto-switching", 
                  justify=tk.CENTER).pack(pady=10)
        
        # Wait for choice
        choice_window.wait_window()
        
        if choice[0] == 'bulk':
            self.load_parameters_from_csv_bulk()
        else:
            self.load_parameters_from_csv_single()

    def load_parameters_from_csv_single(self):
        """Load parameters from exported CSV parameter database (single file)"""
        filename = filedialog.askopenfilename(
            title="Load Parameters from CSV (Single File)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Read the CSV file
            df = pd.read_csv(filename)
            
            if df.empty:
                messagebox.showwarning("Empty File", "The CSV file is empty!")
                return
            
            # Get parameter columns (those starting with 'param_')
            param_columns = [col for col in df.columns if col.startswith('param_')]
            
            if not param_columns:
                messagebox.showwarning("No Parameters", "No parameter columns found in CSV file!")
                return
            
            # Create selection dialog if multiple rows exist
            if len(df) > 1:
                # Create a simple selection window
                select_window = tk.Toplevel(self.root)
                select_window.title("Select Parameters to Load")
                select_window.geometry("500x300")
                
                ttk.Label(select_window, text="Multiple parameter sets found. Select one:").pack(pady=10)
                
                # Create listbox with file names
                listbox = tk.Listbox(select_window)
                for idx, row in df.iterrows():
                    audio_file = row.get('audio_file', f'Row {idx+1}')
                    detected = row.get('detected_events', '?')
                    listbox.insert(tk.END, f"{audio_file} (detected: {detected})")
                listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                
                selected_row = [0]  # Use list for mutable reference
                
                def on_select():
                    selection = listbox.curselection()
                    if selection:
                        selected_row[0] = selection[0]
                        select_window.destroy()
                    else:
                        messagebox.showwarning("No Selection", "Please select a parameter set!")
                
                ttk.Button(select_window, text="Load Selected", command=on_select).pack(pady=10)
                
                # Wait for window to close
                select_window.wait_window()
                selected_idx = selected_row[0]
            else:
                selected_idx = 0
            
            # Extract parameters from selected row
            selected_row_data = df.iloc[selected_idx]
            loaded_params = {}
            
            for col in param_columns:
                # Remove 'param_' prefix to get actual parameter name
                param_name = col.replace('param_', '')
                param_value = selected_row_data[col]
                
                # Handle different parameter types
                if param_name in self.detection_params:
                    # Convert to appropriate type based on current parameter
                    current_value = self.detection_params[param_name]
                    if isinstance(current_value, bool):
                        loaded_params[param_name] = bool(param_value)
                    elif isinstance(current_value, int):
                        loaded_params[param_name] = int(param_value)
                    elif isinstance(current_value, float):
                        loaded_params[param_name] = float(param_value)
                    else:
                        loaded_params[param_name] = param_value
            
            # Update detection parameters
            self.detection_params.update(loaded_params)
            
            # Update GUI widgets
            for param, value in loaded_params.items():
                if param in self.param_widgets:
                    if isinstance(self.param_widgets[param], tk.Scale):
                        self.param_widgets[param].set(value)
                    elif isinstance(self.param_widgets[param], tk.BooleanVar):
                        self.param_widgets[param].set(value)
                    elif isinstance(self.param_widgets[param], ttk.Combobox):
                        self.param_widgets[param].set(value)
            
            # Update analysis
            self.update_analysis()
            
            # Show success message
            audio_file = selected_row_data.get('audio_file', 'Unknown')
            messagebox.showinfo("Parameters Loaded", 
                              f"Parameters loaded from CSV\n"
                              f"Source: {audio_file}\n"
                              f"Loaded {len(loaded_params)} parameters")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load parameters from CSV:\n{str(e)}")
    
    def auto_tune_parameters(self):
        """Simple auto-tuning based on current file"""
        if not self.current_data:
            messagebox.showwarning("No Data", "Load an audio file first!")
            return
        
        if not self.current_ground_truth:
            messagebox.showwarning("No Ground Truth", "Load Excel ground truth file first for auto-tuning!")
            return
        
        messagebox.showinfo("Auto-Tune", "Auto-tuning will test different parameter combinations.\nThis may take a moment...")
        
        target_count = self.current_ground_truth.swallow_count
        best_params = None
        best_score = float('inf')
        
        for percentile in [40, 50, 60, 70, 80]:
            self.detection_params['energy_percentile'] = percentile
            events, _, _, _ = self.detect_swallow_events(
                self.current_data['y_original'], self.current_data['sr'])
            
            score = abs(len(events) - target_count)
            if score < best_score:
                best_score = score
                best_params = {'energy_percentile': percentile}
        
        if best_params:
            self.detection_params.update(best_params)
            self.param_widgets['energy_percentile'].set(best_params['energy_percentile'])
            self.update_analysis()
            messagebox.showinfo("Auto-Tune Complete", 
                              f"Best parameters: {best_params}\nTarget: {target_count}, Score: {best_score}")
        else:
            messagebox.showwarning("Auto-Tune Failed", "Could not find better parameters!")
    
    def run(self):
        """Run the GUI application"""
        self.create_gui()
        
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_reqwidth()) // 2
        y = (self.root.winfo_screenheight() - self.root.winfo_reqheight()) // 2
        self.root.geometry(f"+{x}+{y}")
        
        self.update_spl_storage_info()
        self.update_parameter_storage_info()
        
        self.root.mainloop()

def main():
    """Main function to run the GUI"""
    print("Starting Enhanced Interactive Swallow Detection GUI...")
    print("Features:")
    print("   • Parameter storage with auto-save/load")
    print("   • Enhanced SPL vs Frequency analysis")
    print("   • Excel event durations display")
    print("   • Batch data export capabilities")
    
    app = SwallowDetectorGUI()
    app.run()

if __name__ == "__main__":
    main()