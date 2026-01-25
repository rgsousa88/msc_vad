import os
import sys
import threading
import subprocess

MAX_THREADS = 4

semaphore = threading.Semaphore(MAX_THREADS)

def extract_frames_func(video_file, dest_frames_folder, sample_id):
    ffmpeg_command = f"ffmpeg -hwaccel cuda -i {video_file} -vf \"fps=3\" {os.path.join(dest_frames_folder,sample_id + '_%04d.png')}"
    print(f"Executing {ffmpeg_command}")
    completed_process = subprocess.run(["ffmpeg","-hwaccel","cuda",
                                        "-i", f"{video_file}",
                                        "-vf","fps=3",
                                        f"{os.path.join(dest_frames_folder,sample_id + '_%04d.png')}"],check=True)
    
    return

def extract_frames(path:str):
    video_files = [os.path.join(root,file) for root,dirs,files in os.walk(path) for file in files if file.endswith(".mp4")]
    video_files.sort()
    
    split = ""
    if "train" in path:
        split = "train"
    elif "val" in path:
        split = "val"
    else:
        return
    
    dest_folder_fmt = f"/mnt/c/dataset/MSC/shanghai/{split}_frames"
    for video_file in video_files:
        sample_id = os.path.basename(video_file).replace(".mp4","")
        
        dest_frames_folder = os.path.join(dest_folder_fmt,f"{sample_id}")
        if not os.path.exists(dest_frames_folder):
            os.makedirs(dest_frames_folder)
            print(f"Creating {dest_frames_folder}")

        semaphore.acquire()
        thread = threading.Thread(target=extract_frames_func, args=(video_file, dest_frames_folder, sample_id))
        thread.start()
        thread.join()
        semaphore.release()

if __name__ == "__main__":
    train_mp4_path = "/mnt/c/dataset/MSC/Shanghai/train_mp4"
    val_mp4_path = "/mnt/c/dataset/MSC/shanghai/val_mp4"

    assert os.path.exists(train_mp4_path)
    assert os.path.exists(val_mp4_path)

    extract_frames(train_mp4_path)
    extract_frames(val_mp4_path)