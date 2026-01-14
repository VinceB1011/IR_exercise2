
def get_peaks(Y, k, t):
    size = (2*k + 1, 2*t + 1)
    result = ndimage.maximum_filter(Y, size=size, mode='constant')
    cmap = (Y == result) & (Y > 0.01)
    return np.argwhere(cmap)

def generate_hashes_from_peaks(peaks, dt_min, dt_max, df_max, fan_out):
    # Sort peaks by time (column 1)
    peaks = peaks[peaks[:, 1].argsort()]

    pairs = []

    for i in range(len(peaks)):
        anchor = peaks[i]
        f1, t1 = anchor[0], anchor[1]
        
        # Look for targets in a window after the anchor
        count = 0
        for j in range(i + 1, len(peaks)):
            target = peaks[j]
            f2, t2 = target[0], target[1]
            dt = t2 - t1
            
            # Check if target is within the Target Zone
            if dt_min <= dt <= dt_max and abs(f2 - f1) <= df_max:
                # 1. Create the 32-bit Hash
                # Example: pack f1 (10 bits), f2 (10 bits), and dt (12 bits)

                hash_32 = (f1 | (f2 << 10) | (dt << 20)) & 0xFFFFFFFF

                if i == 0 and count == 0:
                    print(f"    [DEBUG] f1:{f1}, f2:{f2}, dt:{dt} -> Hash:{bin(hash_32)}")
                
                # 2. Store (hash, t1)
                pairs.append((hash_32, t1))
                
                count += 1
                if count >= fan_out:
                    break
    print(f"  > Pairs generated: {len(pairs)} (Avg fan-out: {len(pairs)/len(peaks):.2f})")
    return pairs