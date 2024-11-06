from pytorch_fid.fid_score \
    import calculate_activation_statistics, \
    calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
import os
import random

device = "cuda:1"

# take 2048 random real samples
directory_real = '/home/aolivepe/ECE60146/HW8/data/celeba_dataset_64x64/0'
real_paths = os.listdir(directory_real)
real_paths = random.sample(real_paths, 2048)
real_paths = [os.path.join(directory_real, file) for file in real_paths]

# take the 2048 synthetic samples
directory_fake = '/home/aolivepe/ECE60146/HW8/DLStudio-2.4.3/ExamplesAdversarialLearning/results_mygan_Gskip_Dskip_experiments_final'
fake_paths = os.listdir(directory_fake)
fake_paths = random.sample(fake_paths, 2048)
fake_paths = [os.path.join(directory_fake, file) for file in fake_paths]

dims = 2048

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)
m1, s1 = calculate_activation_statistics(
real_paths, model, device=device)
m2, s2 = calculate_activation_statistics(
fake_paths, model, device=device)
fid_value = calculate_frechet_distance(m1, s1, m2, s2)
print(f'FID: {fid_value:.2f}')