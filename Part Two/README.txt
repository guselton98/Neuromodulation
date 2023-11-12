ISSUES...

matplotlib backend error: 'plt' not defined
Solution: add the following code
    >>> import matplotlib
    >>> matplotlib.use('TkAgg')

Unable to install cv2 library:
Go to terminal and type the following
    > pip3 install opencv-python

Upgrade pip
Go to terminal and type:
    > pip install --upgrade pip
Check version of pip by
    > pip3 --version
The version should be > 23.3.1