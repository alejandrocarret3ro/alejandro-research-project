import os
import subprocess
import random

y4m_dir = r"C:\Users\User\OneDrive\Documents\trinity\Computer Engineering\Year5\Masters Project\xiph_videos"        # Folder with your y4m files
output_dir = r"C:\Users\User\OneDrive\Documents\trinity\Computer Engineering\Year5\Masters Project\xiph_frames"     # Where PNGs will be saved
max_videos = 100
frames_per_video = 30

# Find all y4m files
y4m_files = sorted([f for f in os.listdir(y4m_dir) if f.endswith('.y4m')])
print(f"Found {len(y4m_files)} y4m files")

# Select random subset if needed
random.seed(42)
if len(y4m_files) > max_videos:
    y4m_files = random.sample(y4m_files, max_videos)

for i, filename in enumerate(y4m_files):
    video_name = os.path.splitext(filename)[0]
    video_out = os.path.join(output_dir, f"xiph_{video_name}")
    y4m_path = os.path.join(y4m_dir, filename)

    if os.path.exists(video_out) and len(os.listdir(video_out)) >= 25:
        print(f"[{i+1}/{len(y4m_files)}] {video_name}: already done, skipping")
        continue

    os.makedirs(video_out, exist_ok=True)

    # Get total frame count
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-count_frames', '-select_streams', 'v:0',
         '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', y4m_path],
        capture_output=True, text=True
    )

    try:
        total_frames = int(result.stdout.strip())
    except:
        print(f"[{i+1}/{len(y4m_files)}] {video_name}: can't read frame count, skipping")
        continue

    if total_frames < 30:
        print(f"[{i+1}/{len(y4m_files)}] {video_name}: too short ({total_frames} frames), skipping")
        continue

# Extract ALL frames
    subprocess.run(
        ['ffmpeg', '-y', '-i', y4m_path,
         '-q:v', '1',
         os.path.join(video_out, 'frame_%04d.png')],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    extracted = len([f for f in os.listdir(video_out) if f.endswith('.png')])
    print(f"[{i+1}/{len(y4m_files)}] {video_name}: {extracted} frames from {total_frames} total")

# Summary
total_videos = len([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])
total_frames = sum(
    len([f for f in os.listdir(os.path.join(output_dir, d)) if f.endswith('.png')])
    for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
)
print(f"\nDone! {total_videos} videos, {total_frames} frames in {output_dir}")
print("Now zip this folder and upload to Google Drive")