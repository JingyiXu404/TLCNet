# RGBD NYU v2
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 16 --layer 5 --gpu_id 1 --testset 'NYU'
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 8 --layer 5 --gpu_id 1 --testset 'NYU'
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 4 --layer 5 --gpu_id 1 --testset 'NYU'
# RGBD RGBDD v2
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 16 --layer 5 --gpu_id 1 --testset 'RGBDD'
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 8 --layer 5 --gpu_id 1 --testset 'RGBDD'
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 4 --layer 5 --gpu_id 1 --testset 'RGBDD'
# RGBD Middlebury v2
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 16 --layer 5 --gpu_id 1 --testset 'Middle'
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 8 --layer 5 --gpu_id 1 --testset 'Middle'
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 4 --layer 5 --gpu_id 1 --testset 'Middle'
# RGBD Lu v2
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 16 --layer 5 --gpu_id 1 --testset 'Lu'
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 8 --layer 5 --gpu_id 1 --testset 'Lu'
python test.py --modelname TLCNet --phase test --load_pre 'cpts/' --scale 4 --layer 5 --gpu_id 1 --testset 'Lu'
