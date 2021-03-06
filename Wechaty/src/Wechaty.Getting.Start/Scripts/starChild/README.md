# StarChild / 星星的孩子

Project that is a demo for creating a plaything for autistic children, made for a PaddlePaddle/Wechaty/Mixlab contest.

For more background info, please refer [here](https://www.linkedin.com/pulse/part-1-baby-steps-applied-computer-vision-training-autistic-pan).

Usage: `python run_test [image directory]`

The repo includes some test images under the `test` directory, you can try with them.

## Import Usage

Use below method if you are packaging the core codes as part of an external UX element:

1. `import run_test`
2. Call `run_test.run_evaluation` with two arguments: a) requested shape string* and b) test image local file path. It will return `True` or `False` depending on whether it thinks the image includes the requested shape

*) Shape should be one of `r`, `s` or `t` strings (representing "round", "square" and "triangle" shapes)

Example code snippet in python:

    test_image_path = "/home/path/to/image.jpg"
    test_shape = random.choice(["r", "s", "t"])
    res = run_evaluation(test_shape, test_image_path)
    print("Random shape: {}, test image: {}, match: {}".format(test_shape, test_image_path, res))
