python preprocess_image.py images/corgi_object_2.png

python main.py --image "images/corgi_object_2_rgba.png" --blip --workspace corgi -O --vram_O --iters 5000
python main.py --image "images/corgi_object_2_rgba.png" --blip --workspace corgi -O
python main.py --image "images/corgi_object_2_rgba.png" --blip --workspace corgi -O --vram_O --iters 5000 --backbone grid_tcnn
python main.py --text "a corgi" --workspace corgi-dreamfusion -O
python main.py --text "a DSLR photo of a delicious hamburger" --workspace hamburger5 -O
python main.py --text "A Matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace castle -O --VSD
python main.py --text "a DSLR photo of a delicious croissant" --workspace croissant -O --VSD --vram_O
python main.py -O --image images/corgi_object_2_rgba.png --workspace corgi-zero123 --iters 5000
python main.py -O --image images/corgi_object_2_rgba.png --blip --workspace corgi-zero123-1 --iters 5000 --guidance 'zero123'

python main.py --image "images/corgi_object_2_rgba.png" --blip --workspace corgi -O --vram_O --iters 5000 --test --save_mesh
python main.py --image "images/corgi_object_2_rgba.png" --blip --workspace corgi1 -O --test --save_mesh
python main.py --image "images/corgi_object_2_rgba.png" --blip --workspace corgi -O --vram_O --iters 5000 --backbone grid_tcnn --test --save_mesh
python main.py --text "a corgi" --workspace corgi-dreamfusion -O --vram_O --iters 5000 --backbone grid_tcnn --test --save_mesh
python main.py --text "a DSLR photo of a delicious hamburger" --workspace hamburger -O --save_mesh
python main.py -O --image images/corgi_object_2_rgba.png --workspace corgi-zero123 --iters 5000 --save_mesh
