# CNN

## Why we use CNNs

### 1. Images are Big

Images used for CV problems are often 224x224 or larger. For a color image with RGB color channels, that comes out to 224x224x3 = **150,528** input features! If we had a typical hidden layer in a network with say, 1024 nodes, we'd have to train **150million + weights for the first layer alone**.

And really, we don't need that many weights. The good thing about images is that we know **pixels are most useful in the context of their neighbors** i.e., objects in images are made up of small, *localized* features (like the circular iris of an eye or the square corner of a piece of paper)

### 2. Positions can change

If you train a network to detect dogs, you'd want it to be able to detect a dog *regardless of where it appears in the image*. If you're training a network on images of dogs in a specific position - feeding it dogs in a *different* position would fail to produce the same results!

# What are CNNs?

They're basically just neural networks that use **Convolution layers** (Conv layers), which are based on the mathematical operation of [convolution](https://en.wikipedia.org/wiki/Convolution). Conv layers consist of a set of **filters -** basically just a 2D matrix of numbers. Here's a 3x3 filter:

![](Untitled-445d64a6-d334-4b30-b3b8-cfee83103ccb.png)

We can use an input image and a filter to produce an output image by **convolving** the filter with the input image which involves

1. Overlaying the filter on top of the image at some location.
2. Performing **element-wise multiplication** between the values in the filter and their corresponding values in the image.
3. Summing up all of the element-wise products - this sum is the output value for the **destination pixel** in the output image.
4. Repeating for all locations.

The 4 steps are abstract, so a visualization is to follow. Consider the following 4x4 grayscale image and a 3x3 filter:

![](https://victorzhou.com/media/cnn-post/convolve-example-1.svg)

The numbers in the image represents pixel intensities, where 0 is black and 255 is white. We'll convolve this input image and the filter to produce a 2x2 output image:

![](Untitled-54577d13-9dd4-48d3-85ae-7097908401d1.png)

To start, let's overlay our filter in the top left corner of the image:

![](https://victorzhou.com/media/cnn-post/convolve-example-2.svg)

Next we perform element-wise multiplication between the overlapping image values and filter values. Here are the results, starting from the top left corner, going right, then down:

[Untitled](https://www.notion.so/581f0d45603e4876ba24e1c072a2fde1)

Next we sum up the results: `62 - 33 = 29` . We then place this result in the destination pixel of our output image. Since our filter is overlaid in the top left corner of the input image, our destination pixel is the top left pixel of the output image. 

![](https://victorzhou.com/convolve-output-69b4c1dd078ee363317bb8fa323eaace.gif)

## How is this useful?

Let's zoom out and look at this at a higher level. What does convolving an image with a filter do? We can start by using the example 3x3 filter we've been using, which is known as the vertical [Sobel filter](https://en.wikipedia.org/wiki/Sobel_operator):

![](https://victorzhou.com/media/cnn-post/vertical-sobel.svg)

Here's an example of what it does:

![](https://victorzhou.com/static/44a1ff59f9a2c7f62cf9f56a8398efd0/a8200/lenna%2Bvertical.png)

Or a horizontal Sobel filter:

![](https://victorzhou.com/media/cnn-post/horizontal-sobel.svg)

![](https://victorzhou.com/static/342e8364a392ac2cbcd4ecc7b9aacaa1/a8200/lenna%2Bhorizontal.png)

So what's happening? **Sobel filters are edge-detectors**. The vertical Sobel filter detects vertical edges and the horizontal Sobel filter detects horizontal edges. The output images are now easily interpreted: a bright pixel (one that has a high value) in the output image indicates there's a strong edge around there in the original image.

Think of why an edge-detected image might be more useful than the raw image, specifically for MNIST. A CNN trained on MNIST might look for the digit 1, for example, by using an edge-detection filter and checking for two prominent vertical edges near the center of the image. In general, **convolutions help us look for specific localized image features** (like edges) that we can use later in the network.

## Padding

Above, we convolved a 4x4 input image with a 3x3 filter to produce a 2x2 output image - we'd rather have the output image be the same size of the input image. To do this, we add zeros around the image so that we can overlay the filter in more places. A 3x3 filter requires 1 pixel of padding:

![](https://victorzhou.com/media/cnn-post/padding.svg)

This is called **"same"** **padding**, since the input and output have the same dimensions. Not using any padding is sometimes referred to as **"valid" padding**.

## Conv Layers

Now that we know how image convolution works and why it's useful, let's see how it's actually used in CNNs. As mentioned before, CNNs include **conv layers** that use a set of filters to turn input images into output images. A conv layer's primary parameter is the **number of filters** it has.

For our MNIST CNN, we'll use a small conv layers with 8 filters as the initial layer in our network. This means it'll turn a 28x28 input image into a 26x26x8 (bc valid padding) output **volume**:

![](https://victorzhou.com/media/cnn-post/cnn-dims-1.svg)

## Implementing Convolution

Here's the conv layer's feedforward portion, which takes care of convolving filters with an input image to produce an output volume. For simplicity, we'll assume filters are always 3x3 (not always true):

    import numpy as np
    
    class Conv3x3:
    	# a Convolution layer using 3x3 filters
    
    	def __init__(self, num_filters):
    		self.num_filters = num_filters
    
    		# filters is a 3d array with dimensions (num_filters, 3, 3)
    		# We divide by 9 to reduce the variance of our initial values
    		self.fitlers = np.random.randn(num_filters, 3, 3) / 9

*Dividing by 9 during initialization is v important. If the initial values are too large or too small, training the network will be ineffective. Read about [Xavier Initialization](https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks) to learn more*

Next, the actual convolution:

    class Conv3x3:
    	# ...
    
    	def iterate_regions(self, image):
    		'''
    		Generates all possible 3x3 image regions using valid padding
    			- Image is a 2d numpy array
    		'''
    		h, w = image.shape
    		
    		for i in range(h - 2):
    			for j in range(w - 2):
    				im_region = image[i:(i+3), j:(j+3)]
    				yield im_region, i, j
    
    	def forward(self, input):
    		'''
    		Performs a forward pass of the conv layer using the given input.
    		Returns a 3d numpy array with dimensions (h, w, num_filters)
    			- input is a 2d numpy array
    		'''
    		h, w = input_shape
    		output = np.zeros((h - 2, w - 2, self.num_filters))
    
    		for im_region, i, j in self.iterate_regions(input):
    			output[i, j] = np.sum(im_region * self.filters, axis=(1,2))
    
    		return output

`iterate_regions()` is a helper generator that yields all valid 3x3 image regions for us. This is useful for the backwards portion of this class later on.

The line of code that actually performs the convolutions is `output[i,j]` . Let's break it down.

- We have `im_region` , a 3x3 array containing the relevant image region
- We have `self.filters`, a 3d array
- We do `im_region * self.filters`, which uses numpy broadcasting to perform element-wise multiplication on the two arrays. The result is a 3d array with the same dimensions as `self.filters`.
- We `np.sum()` the result of the previous step using `axis=(1,2)`  which produces a 1d array of length `num_filters` where each element contains the convolution result for the corresponding filter.
- We assign the result to `output[i,j]` which contains convolution results for pixel `(i,j)` in the output

This sequence is performed for each pixel in the output until we obtain our final output volume.

## Pooling

Neighboring pixels in images tend to have similar values, so conv layers will typically produce similar values for neighboring pixels in outputs. As a result, **much of the information contained in a conv layer's output is redundant**. 1 pixel shifted from the original one is probably going to react to the same thing as the original.

Pooling layers solve this problem. All they do is reduce the size of the input it's given by pooling values together in the input. Pooling is usually done by a simple operation like `max` , `min` , or `average` . Here's an example of a Max Pooling layer with a pooling size of two:

![](https://victorzhou.com/pool-ac441205fd06dc037b3db2dbf05660f7.gif)

To perform *max* pooling, we traverse the input image in 2x2 blocks (bc pool size=2) and put the *max* value into the output image at the corresponding pixel. That's it.

**Pooling dives the input's width and height by the pool size**. Ex, for MNIST CNN, we'll place a Max Pooling layer with a pool size of 2 right after our initial conv layer. The pooling layer will transform a 26x26x8 input into a 13x13x8 output.

### Implementing Pooling

    import numpy as np
    
    class MaxPool2:
      # A Max Pooling layer using a pool size of 2.
    
      def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2
    
        for i in range(new_h):
          for j in range(new_w):
            im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
            yield im_region, i, j
    
      def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))
    
        for im_region, i, j in self.iterate_regions(input):
          output[i, j] = np.amax(im_region, axis=(0, 1))return output

The only difference between this and the `Conv3x3` class is the `output[i,j]` line. To find the max from a given image region, we use `np.amax()`, numpy's array max method. We then set `axis=(0, 1)`, because we only want to maximize over the first two dimensions, height and width, not the third - `num-filters` 

## Softmax

To complete the CNN, we need to give it the ability to actually make predictions. We can do that by using the standard final layer fora multiclass classification problem: the **Softmax** layer - a fully-connected (dense) layer that uses the softmax activation function.

- fully-connected layers have every node connected to every output from the previous layer.

**Softmax turns arbitrary real values into *probabilities***. The math is pretty simple: given some numbers

1. Raise *e* to the power of each of those numbers
2. Sum up all the exponentials (powers of *e*) - this result is the denominator
3. Use each number's exponential as its numerator
4. Probability = Numerator/Denominator

The outputs of the Softmax transform are always in the range [0, 1] and add up to 1. Hence, they're **probabilities**.

Here's an example using the numbers -1, 0, 3, and 5:

$$Denominator  = e^{-1} + e^0 + e^3 + e^5 = 169.87$$

[Untitled](https://www.notion.so/81b6f8d417cc40929534782c8de4dda1)

### Usage

We'll use a softmax layer with **10 nodes, one for each digit**, as the final layer in our CNN. Each node in the layer will be connected to every input. After the softmax transformation is applied, **the digit represented by the node with the highest probability** will be the output of the CNN!

![](https://victorzhou.com/media/cnn-post/cnn-dims-3.svg)

### Cross-Entropy Loss

Why bother transforming the outputs into probabilities? Won't the highest output value always have the highest probability? **Correct - we don't actually need to use softmax to predict a digit** - we could just pick the digit with the highest output from the network.

What softmax really does is helps us **quantify how sure we are of our prediction**, which is useful when training and evaluating our CNN. More specifically, using softmax lets us use **cross-entropy loss**, which takes into account how sure we are of each prediction. Here's how we calculate cross-entropy loss:

$$L = -ln(p_c)$$

where *c* is the correct class and *p_c* is the predicted probability for class *c.* The lower the loss the better.

### Implementing Softmax

    import numpy as np
    
    class Softmax:
    	# A standard fully-connected layer with softmax activation
    
    	def __init__(self, input_len, nodes):
    		# We divide the input_len to reduce the variance of our initial values
    		self.weights = np.random.randn(input_len, nodes) / input_len
    		self.biases = np.zeros(nodes)
    
    	def forward(self, input):
    		"""
    		Performs a forward pass of the softmax layer using the given input
    		Returns a 1d numpy array containing the respective probability values.
    			- input can be any array with dimensions
    		"""
    		input = input.flatten()
    		
    		input_len, nodes = self.weights.shape
    
    		totals = np.dot(input, self.weights) + self.biases
    		exp = np.exp(totals)
    		return exp / np.sum(exp, axis=0)

- We `flatten()` the input to make it easier to work with since we no longer need its shape
- `np.dot()` multiplies `input` and `self.weights` element-wise and sums the results
- `np.exp()` calculates the exponentials used for Softmax.

# Part 2 - Training

Training a NN consists of two phases

1. A **forward** phase, where the input is passed completely through the network.
2. A **backward** phase, where gradients are backpropagated and weights are updated.

For this walkthrough, we'll follow these steps but implement two specific ideas:

1. During the forward phase, each layer will cache any data (inputs, intermediate values) that it'll need for the backward phase. This means that any backward phase is followed by a forward phase
2. During the backward phase, each layer will **receive a gradient** and **return a gradient**. It will receive the gradient of loss with respect to its *outputs dL/d_out* and return the gradient of loss with respect to its *inputs* dL/d_in

Caching:

    class Softmax:
    	# ...
    	def forward(self, input):
    		self.last_input_shape = input.shape #new
    		
    		input = input.flatten()
    		self.last_input = input #new
    		
    		input_len, nodes = self.weights.shape
    
    		totals = np.dot(input, self.weights) + self.biases
    		self.last_totals = totals #new
    
    		exp = np.exp(totals)
    		return exp / np.sum(exp, axis=0)

We cache 3 useful things for backprop:

1. The `input`'s shape *before* we flatten it
2. The `input` after we flatten it
3. The **totals**, which are the values passed into the softmax activation.

## Backprop: Softmax

We start our way from the end and work our way towards the beginning - thats how backprop works.

The first thing we need to calculate is the input to the Softmax layer's backwards phase (dL/dout), where *out* is the output from the Softmax layer: a vector of 10 probabilities. One fact we can use about this is that *it's only nonzero for c, the correct class*. That means we can ignore everything but *out_s(c)*. There's more calculus here but I'm not good enough with latex.

    class Softmax:
    	#...
    	def backprop(self, d_L_d_out):
    		"""
    		Performs a backward pass of the softmax layer
    		Returns the loss gradient for this layer's inputs
    		"""
    		# We know only 1 element of d_L_d_out will be nonzero
    		for i, gradient in enumerate(d_L_d_out):
    			if gradient == 0:
    				continue
    
    			#e^totals
    			t_exp = np.exp(self.last_totals)
    
    			# Sum of all e^totals
    			S = np.sum(t_exp)
    
    			# Gradients of out[i] against totals
    			d_out_d_t = -t_exp[i] * t_exp / (S**2)
    			d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S**2)
    			
    		# ...

We ultimately want the gradients of loss against weights, biases, and input:

- We'll use the weights gradient, *dL/de,* to update our layer's weights
- We'll use the biases gradient, *dL/db*, to update our layer's biases
- We'll return the input gradient, *dL/dinput*, from our `backprop()` method so the next layer can use it. This is the return gradient we talked about in the Training Overview section.

To calculate those 3 loss gradients, we first need to derive 3 more results: the gradients of *totals* against weights, biases, and input. The relevant equation is:

$$t = w * input + b$$

Putting this into the code...

    class Softmax:
    	# ...
    	def backprop():
    		# ...
    
    		# Gradients of totals against weights/biases/input
    		d_t_d_w = self.last_input
    		d_t_d_b = 1
    		d_t_d_inputs = self.weights
    		
    		# Gradients of loss against totals
    		d_L_d_t = gradient * d_out_d_t
    
    		# Gradients of loss against weights/biases/input
    		d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
    		d_L_d_b = d_L_d_t * d_t_d_b
    		d_L_d_inputs = d_t_d_inputs @ d_L_d_t

First, we pre-calculate `d_L_d_t` since we'll use it several times. Then, we calculate each gradient:

- **`d_L_d_w`**: We need 2d arrays to do matrix multiplication (`@`), but `d_t_d_w` and `d_L_d_t` are 1d arrays. `np.newaxis` lets us easily create a new axis of length one, so we end up multiplying matrices with dimensions (`input_len`, 1) and (`, `nodes`). Thus, the final results for `d_L_d_w` will have shape (`input_len`, `nodes`), which is the same as `self.weights`.
- **`d_L_d_b`:** This one is straightforward, since `d_t_d_b` is 1.
- **`d_L_d_inputs`**: We multiply matrices with dimensions (`input_len`, `nodes`) and (`nodes`, 1) to get a result with length `input_len`.

All that's left is to actually train the Softmax layer. We'll update the weights and bias using SGD and then return `d_L_d_inputs`

    class Softmax:
    	# ...
    	def backprop(self, d_L_d_out, learn_rate):
    		# ...
    		
    		# update weights / biases
    		self.weights -= learn_rate * d_L_d_w
    		self.biases -= learn_rate * d_L_d_b
    
    		return d_L_d_inputs.reshape(self.last_input_shape)

Notice that we added a `learn_rate` parameter that controls how fast we update our weights. Also, we have to `reshape()` before returning `d_L_d_inputs` because we flattened the input during our forward pass:

    class Softmax:
    	# ...
    	def forward(self, input):
    		# ...
    		input = input.flatten()
    		self.last_input = input

Reshaping to `last_input_shape` ensures that this layer returns gradients for its input in the same format that the input was originally given to it.

## Backprop: Max Pooling

A **Max Pooling layer** can't be trained because it doesn't actually have any weights, but we still need to implement a `backprop()` method for it to calculate gradients. We'll start by adding forward phase caching again. All we need to cache this point in time is the input:

    class MaxPool2:
    	# ...
    	
    	def forward(self, input):
    		'''
    		Performs a forward pass of the maxpool alyer using the given input
    		Returns a 3d numpy array with dimensions (h / 2, w/ 2, num_filters)
    		- input is a 3d numpy array with dimensions (h, w, num_filters)
    		'''
    		self.last_input = input

During the forward pass, the Max Pooling layer takes an input volume and halves its width and height dimensions by picking the max values over 2x2 blocks. The backward pass does the opposite: **we'll double the width and height** of the loss gradient by assigning each gradient value to **where the original max value was** in its corresponding 2x2 block. Forward:

![](https://victorzhou.com/media/cnn-post/maxpool-forward.svg)

Backward:

![](https://victorzhou.com/media/cnn-post/maxpool-backprop.svg)

Each gradient value is assigned to where the original max value was and every other value is zero.

Why does the backward phase for a Max Pooling layer work like this? Think about what *dL/dinputs* intuitively should be. An input pixel that isn't the max value in its 2x2 block would have *zero marginal effect on the loss* because changing that value slightly wouldn't change the output at all! In other words, *dL/dinput* = 0 for non-max pixels. On the other hand, an input pixel that *is* the max value, would have its value passed through to the output, so *doutput/dinput* = 1, meaning *dL/dinput = dL/doutput*.

We can do this with the same `iterate_regions` from above:

    class MaxPool2:
    	# ...
    	def iterate_regions(self, image):
    		'''
    		Generates non-overlapping 2x2 image regions to pool over
    		- image is a 2d numpy array
    		'''
    		h, w, _ image.shape
    		new_h = h // 2
    		new_w = w // 2
    
    		for i in range(new_h):
    			for j in range(new_w):
    				im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
    				yield im_region, i, j
    
    	def backprop(self, d_L_d_out):
    		'''
    		Performs a backward pass of the maxpool layer
    		Returns the loss gradient for this layer's inputs
    		- d_L_d_out is the loss gradient for this layer's outputs
    		'''
    		d_L_d_input = np.zeros(self.last_input_shape)
    
    		for im_region, i, j in self.iterate_regions(self.last_input):
    			h, w, f = im_region.shape
    			amax = np.amax(im_region, axis=(0, 1))
    			
    			for i2 in range(h):
    				for j2 in range(w):
    					for f2 in range(f):
    						# If this pixel was the max value, copy the gradient to it
    						if im_region[i2, j2, f2] == amax[f2]:
    							d_L_d_input[i * 2 + i2, j *2 + j2, f2] = d_L_d_out[i, j, f2]
    		
    		return dL_d_input

For each pixel in each 2x2 image region in each filter, we copy the gradient from `d_L_d_out` to `d_L_d_input` if it was the max value during the forward pass.

## Backprop: Conv

The final layer! Backpropogating through a Conv layer is the core of training a CNN. The forward phase caching is simple:

    class Conv3x3:
    	# ...
    	
    	def forward(self, input):
    		'''
    		Performs a forward pass of the conv layer using the given input
    		Returns a 3d numpy array with dimensions (h, w, num_filters)
    		- input is a 2d numpy array
    		'''
    		self.last_input = input

*Reminder about our implementation: For simplicity, **we assume the input to our conv layer is a 2d array**. This only works for us because we use it as the first layer in our network. If we were building a bigger network that needed to use `Conv3x3` multiple times, we'd have to make the input a **3d** array.*

We're primarily interested in the loss gradient for the filters in our conv layer, since we need that to update our filter weights. We already have *dl/dout* for the conv layer, so we just need *dout/dfilters*. To calculate that, we ask ourselves this: how would changing a filter's weight affect the conv layer's output?

The reality is that **changing any filter weights would affect the *entire* output image** for that filter, since *every* output pixel uses *every* pixel weight during convolution. To make this even easier to think about, let's just think about one output pixel at a time: **how would modifying a filter change the output of *one* specific output pixel.** An example with a 3x3 image (left) convolved with a 3x3 filter (middle) to produce a 1x1 output (right):

![](https://victorzhou.com/media/cnn-post/conv-gradient-example-1.svg)

We have a 3x3 image convolved with a 3x3 filter of all zeros to produce a 1x1 output. What if we increased the center filter weight by 1? The output would increase by the center image value, 80:

![](https://victorzhou.com/media/cnn-post/conv-gradient-example-2.svg)

Similarly, increasing any of the other filter weights by 1 would increase the output by the value of the corresponding image pixel! This suggests that the derivative of a specific output pixel with respect to a specific filter weight is just the corresponding image pixel value (some more calculus here).

We can now implement backprop:

    class Conv3x3:
    	# ...
    	def backprop(self, d_L_d_out, learn_rate):
    		'''
    		Performs a backward pass of the conv layer
    		- d_L_d_out is the loss gradient for this layer's outputs
    		- learn_rate is a float
    		'''
    		d_L_d_filters = np.zeros(self.filters.shape)
    
    		for im_region, i, j, in self.iterate_regions(self.last_input):
    			for f in range(self.num_filters):
    				d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
    
    
    		# Update filters
    		self.filters -= learn_rate * d_L_d_filters
    
    		# We don't return anything here since we use Conv3x3 as the first layer
    		# Otherwise, we'd just need to return the loss gradient for this layer's
    		# inputs, just like every other layer in our CNN
    		return None

We apply our derived equation by iterating over every image region / filter and incrementally building the loss gradients. Once we've covered everything, we update `self.filters` using SGD just as before.