import argparse
import os

def parse_scenes(file_path):
    shots = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                start = int(parts[0])
                end = int(parts[1])
                shots.append([start, end])
    return shots

def save_scenes(shots, file_path):
    with open(file_path, 'w') as f:
        for start, end in shots:
            f.write(f"{start} {end}\n")

def merge_shots(shots, min_frames):
    i = 0
    while i < len(shots):
        start, end = shots[i]
        duration = end - start + 1
        
        if duration < min_frames:
            # Need to merge
            if len(shots) == 1:
                break # Cannot merge if only one shot
            
            if i == 0:
                # Merge with next
                next_start, next_end = shots[i+1]
                # New shot is start to next_end
                shots[i] = [start, next_end]
                shots.pop(i+1)
                # Recheck this index
                continue
            elif i == len(shots) - 1:
                # Merge with previous
                prev_start, prev_end = shots[i-1]
                shots[i-1] = [prev_start, end]
                shots.pop(i)
                # Finished with this one, loop will end or we are at end
                break
            else:
                # Merge with shorter neighbor
                prev_start, prev_end = shots[i-1]
                next_start, next_end = shots[i+1]
                
                prev_dur = prev_end - prev_start + 1
                next_dur = next_end - next_start + 1
                
                if prev_dur < next_dur:
                    # Merge with previous
                    shots[i-1] = [prev_start, end]
                    shots.pop(i)
                    # i now points to what was i+1. 
                    # We need to check it (it hasn't been checked yet if we are moving forward)
                    # But wait, if we merge with previous, the previous shot grows.
                    # The current shot `i` is removed.
                    # The loop continues with the same `i`, which is now the next shot.
                    # So we don't increment i.
                    continue
                else:
                    # Merge with next
                    shots[i] = [start, next_end]
                    shots.pop(i+1)
                    # Recheck this index
                    continue
        else:
            i += 1
    return shots

def main():
    parser = argparse.ArgumentParser(description="Merge short shots in a scenes file.")
    parser.add_argument("input_file", help="Path to the input shot_scenes.txt file")
    parser.add_argument("--output_file", help="Path to the output file. Defaults to input_file_merged.txt", default=None)
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second (default: 2.0)")
    parser.add_argument("--min_duration", type=float, default=10.0, help="Minimum duration in seconds (default: 10.0)")
    
    args = parser.parse_args()
    
    min_frames = int(args.fps * args.min_duration)
    print(f"Merging shots shorter than {args.min_duration}s ({min_frames} frames) assuming {args.fps} FPS.")
    
    shots = parse_scenes(args.input_file)
    print(f"Loaded {len(shots)} shots.")
    
    merged_shots = merge_shots(shots, min_frames)
    print(f"Merged into {len(merged_shots)} shots.")
    
    output_path = args.output_file if args.output_file else args.input_file.replace(".txt", "_merged.txt")
    save_scenes(merged_shots, output_path)
    print(f"Saved merged shots to {output_path}")

if __name__ == "__main__":
    main()
