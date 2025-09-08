#rm -v circuits/c=end2end*

python generate_erasure.py

python generate_init.py 3 11
#python generate_init.py 3 13
#python generate_init.py 3 15
#
python generate_init.py 5 11
#python generate_init.py 5 13
#python generate_init.py 5 15

python generate_4q_code.py

