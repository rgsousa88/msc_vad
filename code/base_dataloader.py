from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

image_dir = "/mnt/c/dataset/archive/Fruits/valid"
max_batch_size = 8

@pipeline_def(num_threads=1, device_id=0)
def simple_pipeline(path: str):
    images, labels = fn.readers.file(file_root=path)
    images = fn.decoders.image(images, output_type=types.RGB, name='image')
    images = fn.resize(images.gpu(), resize_x=224, resize_y=224, device='gpu')
    images = fn.normalize(images)

    return images, labels.gpu()

@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0)
def video_seq_pipe(file_root, shape=(224,224), train=True, device='gpu', sequence_length=10, fstride=2, cstride=20, change_color_prob=0.3):
    video = fn.readers.sequence(file_root=file_root,
                                sequence_length=sequence_length,
                                stride=fstride,
                                step=cstride,
                                name='seq')
    
    video = fn.resize(video.gpu(), size=shape, device=device)
    
    if train:
        mirror = fn.random.choice(2)
    else:
        change_color_prob = 0.0
        mirror = 0
    
    video = fn.crop_mirror_normalize(video, dtype=types.FLOAT, std=[255.0], mirror=mirror, output_layout="FHWC")       
    
    do_color_changes = fn.random.coin_flip(probability=change_color_prob, dtype=types.DALIDataType.BOOL)

    if do_color_changes:
        randBrightness = fn.uniform(range=(0.5,1.5))
        randContrast = fn.uniform(range=(0.5,1.5))
        video = fn.brightness_contrast(video, brightness=randBrightness, contrast=randContrast)
    
    video = fn.transpose(video[1:,:,:,:], perm=[3,0,1,2])

    return video

@pipeline_def(num_threads=4, enable_conditionals=True, device_id=0)
def video_pipe(file_root, shape=(224,224), train=True, device='gpu', sequence_length=10, stride=2, step=10, change_color_prob=0.3, initial_pref=512):
    video, labels = fn.readers.video(device=device,
                                    file_root=file_root,
                                    sequence_length=sequence_length,
                                    stride=stride,
                                    step=step,
                                    random_shuffle=True,
                                    initial_fill=initial_pref, name='seq')
    
    video = fn.resize(video.gpu(), size=shape, device=device)
    
    if train:
        mirror = fn.random.choice(2)
    else:
        change_color_prob = 0.0
        mirror = 0
    
    video = fn.crop_mirror_normalize(video, dtype=types.FLOAT, std=[255.0], mirror=mirror, output_layout="FHWC")       
    
    do_color_changes = fn.random.coin_flip(probability=change_color_prob, dtype=types.DALIDataType.BOOL)

    if do_color_changes:
        randBrightness = fn.uniform(range=(0.5,1.5))
        randContrast = fn.uniform(range=(0.5,1.5))
        video = fn.brightness_contrast(video, brightness=randBrightness, contrast=randContrast)
    
    video = fn.transpose(video[1:,:,:,:], perm=[3,0,1,2])

    labels = fn.cast(labels, dtype=types.INT64)

    return video, labels.gpu()

if __name__ == "__main__":
    pipe = simple_pipeline(image_dir, batch_size=max_batch_size, num_threads=1, device_id=0)
    pipe.build()



    

