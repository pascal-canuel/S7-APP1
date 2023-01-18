# run test
`python -m unittest discover . '*test.py'`

# train network
`python train_mnist.py --learning_rate=0.01 --batch_size=10 --epoch_count=1000 --output_path=./checkpoint_v1`

# evaluate a pretrained network
`python train_mnist.py --learning_rate=0.01 --batch_size=10 --epoch_count=1000 --output_path=./checkpoint_v1 --checkpoint_path=./checkpoint_v1/checkpoint_epoch_37.pkl`