**Photo Filter Removal Using Neural Networks**  

```bash
# start training
python Code.py -degin 3 degout 3

# start test - use the below command to test out the image outputs; the output images will be saved in a folder Test_Output
python Code.py -degin 3 degout 3 --regen ./checkpoint.pth

# to check for any other test images; places-instagram > test-list.txt : you can add any images of your choice from the main test-list-main.txt to the test-list.txt file and test the output.

# restored folder has all the test images output.
```