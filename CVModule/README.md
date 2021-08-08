# Computer Vision Module for StarChild

Project that is a demo for creating a plaything for autistic children, made for a PaddlePaddle/Wechaty/Mixlab contest.

For more background info, please refer [here](https://www.linkedin.com/pulse/part-1-baby-steps-applied-computer-vision-training-autistic-pan).

## Server Method Usage

Use below method if you want to set the service up as HTTP API and call it using POST:

1. Run `server/server.py`
2. It will spawn an HTTP server on port 1337 at your localhost
3. You can then call this server with a HTTP POST request in the following way:

### Input json sample

    {
        'gt': str,      # One of 'r', 's' or 't'*
        'img_b64': str, # base64 string of the image binary data
    }

*) Shape should be one of `r`, `s` or `t` strings (representing "round", "square" and "triangle" shapes)

### HTTP response sample

    {
        "status": 0,    # 0 means success, else error
        "message": None # True/False if status 0, else error message
    }

The `true/false` boolean value in the successful return message denotes whether the image matches the shape given in the `gt` or ground truth value. If status is not successful, message will display what the error message was.

## Import Method Usage

Use below method if you are packaging the core codes as part of an external UX element:

1. `import starchild`
2. Use method `starchild.run_evaluation` with two arguments: a) requested shape string* and b) test image local file path. It will return `True` or `False` depending on whether it thinks the image includes the requested shape

*) Shape should be one of `r`, `s` or `t` strings (representing "round", "square" and "triangle" shapes)

Example code snippet in python:

    test_image_path = "/home/path/to/image.jpg"
    test_shape = random.choice(["r", "s", "t"])
    res = run_evaluation(test_shape, test_image_path)
    print("Random shape: {}, test image: {}, match: {}".format(test_shape, test_image_path, res))

## Test Usage

Run: `python starchild [image directory]`

The repo includes some test images under the `test` directory, you can try with them.