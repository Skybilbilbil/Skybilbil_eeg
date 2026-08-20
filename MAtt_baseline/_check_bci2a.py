import pickle
with open(r'D:\EEG_Analysis\Riemannian_Trajectories\saved_data\preprocessed_data_bci2a_raw_ica.pkl', 'rb') as f:
    data = pickle.load(f)
meta = data['subject_metadata']
print(f'Number of subjects: {len(meta)}')
print(f'Metadata keys: {list(meta[0].keys()) if meta else []}')
print(f'Sample frequency: {meta[0].get("sfreq", "N/A")}')
x = data['aligned_left_epochs_list'][0]
print(f'First subject left data shape: {x.shape}')
print(f'All subjects shapes:')
for i in range(len(meta)):
    L = data['aligned_left_epochs_list'][i]
    R = data['aligned_right_epochs_list'][i]
    F = data['aligned_feet_epochs_list'][i]
    H = data['aligned_hands_epochs_list'][i]
    Ls = f'L{list(L.shape)}' if L is not None else 'None'
    Rs = f'R{list(R.shape)}' if R is not None else 'None'
    Fs = f'F{list(F.shape)}' if F is not None else 'None'
    Hs = f'H{list(H.shape)}' if H is not None else 'None'
    print(f'  S{i+1}: {Ls} {Rs} {Fs} {Hs}')