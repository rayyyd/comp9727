import numpy as np

# --- 1. Define two small "TF–IDF" vectors (length = 5) ---
# Imagine these are the TF–IDF scores for “profile” and one “song”
profile = np.array([0.0, 0.2, 0.5, 0.0, 0.1])
song    = np.array([0.1, 0.0, 0.4, 0.3, 0.0])

# --- 2. Binarize (present = True, absent = False) ---
profile_bin = profile > 0
song_bin    = song    > 0

print("Profile binarized:", profile_bin)
print("Song    binarized:", song_bin)

# --- 3. Compute intersection & union masks ---
and_mask = np.logical_and(profile_bin, song_bin)
or_mask  = np.logical_or (profile_bin, song_bin)

print("AND mask:", and_mask)
print(" OR mask:", or_mask)

# --- 4. Count and compute Jaccard similarity ---
intersection = and_mask.sum()   # count of True in AND
union        = or_mask.sum()    # count of True in OR

jaccard_sim = intersection / union
print(f"\nIntersection = {intersection}, Union = {union}")
print(f"Jaccard similarity = {jaccard_sim:.3f}")
