# Build TinySAM yourself: Windows 10, VS Code, and Colab

This is a hint-first study plan for implementing the project yourself. The
finished project beside this guide is a reference, not a file to copy blindly.
Implement one small piece, run its test, and only then read the reference file
if your test passes or you are fully stuck. The hints reveal more detail in
order; stop as soon as one is enough.

Your laptop's i5-7440HQ and GeForce 930MX (2 GB) are enough for the warm-ups,
tests, a small NumPy run, and the demo. **Ignore the Nvidia GPU for training**:
install CPU-only PyTorch. Use free Google Colab for the larger 256-pixel
PyTorch run. The Colab GPU timings have not been measured.

All commands below assume PowerShell is opened at the repository root unless
the step says otherwise. In PowerShell, both `/` and `\` work in many Python
paths; the examples use Windows backslashes.

## 1. Install tools (Windows 10)

1. Install **Python 3.11, 64-bit** from [python.org](https://www.python.org/downloads/windows/).
   On the installer’s first screen, tick **Add python.exe to PATH**. Open a
   *new* PowerShell window and check:

   ```powershell
   py --version
   ```

   If several Python versions are installed, the project commands explicitly
   select 3.11 with `py -3.11`.

2. Install [Git for Windows](https://git-scm.com/download/win), then verify:

   ```powershell
   git --version
   ```

3. Install [Visual Studio Code](https://code.visualstudio.com/).

4. Install these extensions either way:

   - **Extensions panel:** click the blocks icon, search for each extension
     name, and click **Install**.
   - **PowerShell:** run the following from any folder where the `code`
     command is on PATH:

   ```powershell
   code --install-extension ms-python.python
   code --install-extension ms-python.vscode-pylance
   code --install-extension ms-toolsai.jupyter
   ```

   The extensions are Python (`ms-python.python`), Pylance
   (`ms-python.vscode-pylance`), and Jupyter (`ms-toolsai.jupyter`).

## 2. Get the code and make a Python environment

Fork `Mehdi-blip-stack/EfficientSAM` on GitHub if you want your own remote.
Then clone the repository and check out the guide's branch:

```powershell
git clone https://github.com/Mehdi-blip-stack/EfficientSAM.git
cd EfficientSAM
git checkout devin/1792273248-from-scratch-tinysam
```

If you cloned your own fork and that branch is not present there yet, fetch it
from the original repository:

```powershell
git remote add upstream https://github.com/Mehdi-blip-stack/EfficientSAM.git
git fetch upstream devin/1792273248-from-scratch-tinysam
git checkout -b devin/1792273248-from-scratch-tinysam FETCH_HEAD
```

After the pull request is merged, switch to the updated `main` branch instead:

```powershell
git checkout main
git pull origin main
```

Create and activate a virtual environment in the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow locally created scripts for your
Windows user, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Upgrade pip, install the **CPU-only** PyTorch wheel first, then install the
project requirements:

```powershell
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r from_scratch\requirements.txt
```

Do not install CUDA or try to use the 930MX for this project. Check that the
environment is active and Torch sees no CUDA device:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False` for CUDA is expected. In VS Code, open the repository folder, press
`Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select
`.venv\Scripts\python.exe`. Open the **Testing** panel, choose **Configure
Python Tests**, select **pytest**, and choose `from_scratch\tests`. To make the
pytest root directory explicit in workspace settings, use:

```json
"python.testing.pytestArgs": ["--rootdir=from_scratch", "from_scratch/tests"]
```

Use a PowerShell integrated terminal (Terminal → New Terminal, then select
PowerShell from its profile menu). From the repository root, the full reference
suite is:

```powershell
py -3.11 -m pytest from_scratch\tests -q
```

### Keep your implementation separate

Do not overwrite the finished reference package while learning. At the
repository root, create a separate workspace such as:

```text
my_tinysam\
  __init__.py
  tinysam\
    numpy_impl\
    torch_impl\
  tests\
```

Write your implementation under `my_tinysam\tinysam\`. To test your own
NumPy layers with a repository test, copy a relevant test file into
`my_tinysam\tests\`, then change its imports from `tinysam...` to
`my_tinysam.tinysam...`. For example:

```powershell
New-Item -ItemType Directory -Force my_tinysam\tests
Copy-Item from_scratch\tests\test_numpy_gradcheck.py my_tinysam\tests\
```

Change the imports in that copy, but leave the reference test untouched. Run
pytest as a Python module from the repository root so that the root is
importable:

```powershell
py -3.11 -m pytest my_tinysam\tests\test_numpy_gradcheck.py `
  -k test_every_numpy_layer_input_and_parameter_gradients -q
```

This isolated-copy/import approach was checked on Linux with the equivalent
`python -m pytest` invocation. You can also start with one copied test function
and edit its import before implementing the rest of the package.

## 3. Part 0 warm-up: perceptron, KNN, PCA/SVD

These short NumPy and Matplotlib programs review the midterm topics and preview
TinySAM's linear maps, backpropagation, similarities, and projections. Build
each exercise in your own `my_tinysam` workspace before looking at
`from_scratch\warmup\`.

### 3.1 Perceptron and logistic regression

Build a seeded, separable two-class example and plot its separating line and
mistakes per epoch. For labels `y ∈ {-1,+1}`, predict
`sign(wᵀx+b)`. On a mistake where `y(wᵀx+b) ≤ 0`, update
`w ← w + ηyx` and `b ← b + ηy`. Then train logistic regression on the same
points: with `z=Xw+b`, `σ(z)=1/(1+e⁻ᶻ)`, binary-cross-entropy gives
`∂BCE/∂z = σ(z)−y`; therefore `dW=Xᵀdz/N` and `db=Σdz/N`. This is the
single-output version of the gradient flow through TinySAM's `Linear` layer.

Hints:

<details><summary>Hint 1</summary> Begin with two-dimensional points and labels ±1. Make the two classes separable by placing them on opposite sides of a line. </details>

<details><summary>Hint 2</summary> Shuffle sample order each perceptron epoch. Count mistakes before applying updates and stop when an epoch has none. </details>

<details><summary>Hint 3</summary> Convert labels to 0/1 only for logistic BCE. Average the `Xᵀdz` and `Σdz` gradients by the batch size; compare the boundary and the mistakes plot. </details>

Test and reference:

```powershell
py -3.11 -m pytest from_scratch\tests\test_warmup.py `
  -k test_perceptron_learns_separable_data_without_mistakes -q
```

Only after it passes (or if you are fully stuck), compare with
`from_scratch\warmup\perceptron.py`.

### 3.2 K-nearest neighbors

Build vectorized KNN for the three seeded Gaussian blobs. For query `q` and
reference `r`, expand squared Euclidean distance:
`||q-r||² = ||q||² + ||r||² − 2q·r`. Select the nearest `k` with
`np.argpartition`; use a majority vote and resolve equal vote counts by the
smallest class label. Plot accuracy for odd `k` values and decision regions for
the best one. The dot products are a useful bridge to attention's query-key
similarities (attention adds scaling and softmax normalization).

Hints:

<details><summary>Hint 1</summary> Compute one `n_query × n_train` distance matrix. Avoid a Python loop over query/reference pairs. </details>

<details><summary>Hint 2</summary> `np.argpartition(distances, kth=k-1, axis=1)[:, :k]` gives an unordered nearest-neighbor set. Ordering those `k` entries is unnecessary for voting. </details>

<details><summary>Hint 3</summary> Use `np.unique(..., return_counts=True)` on the neighbor labels. `np.unique` returns sorted labels; selecting the first label tied at the maximum count implements the smallest-label rule. </details>

Test and reference:

```powershell
py -3.11 -m pytest from_scratch\tests\test_warmup.py `
  -k test_knn_distances_match_brute_force_and_training_one_nn_is_exact -q
py -3.11 -m pytest from_scratch\tests\test_warmup.py `
  -k test_knn_vote_ties_choose_smallest_label -q
```

Only after the tests pass (or if you are fully stuck), compare with
`from_scratch\warmup\knn.py`.

### 3.3 PCA by SVD and power iteration

Center `X` by subtracting its feature mean. From `X=UΣVᵀ`, the columns of
`V` are principal directions and covariance eigenvalues are
`λᵢ=σᵢ²/(n−1)`. Independently estimate directions from
`C=XᵀX/(n−1)` by power iteration
`v ← Cv/||Cv||`; record `λ=vᵀCv` and deflate
`C ← C−λvvᵀ`. Eigenvectors may differ by sign, so compare their absolute
cosine similarity. Reconstruct with `μ+(X−μ)VₖVₖᵀ`; the error should not
increase as `k` grows. This connects PCA/SVD to learned projections and
low-rank matrix factorization in TinySAM.

Hints:

<details><summary>Hint 1</summary> Use the centered data for both methods. The sample covariance denominator is `n−1`; singular-vector signs are arbitrary. </details>

<details><summary>Hint 2</summary> Normalize every power-iteration update. Deflate with the eigenvalue estimate and the outer product `vvᵀ` before finding the next component. </details>

<details><summary>Hint 3</summary> Project into the first `k` directions, then project back and add the mean. For the optional Pets images, resize each RGB image to 32×32 grayscale before flattening; compare reconstructions for `k=5,20,50`. </details>

Tests and reference:

```powershell
py -3.11 -m pytest from_scratch\tests\test_warmup.py `
  -k test_power_iteration_matches_svd_and_reconstruction_error_falls -q
python from_scratch\warmup\pca_svd.py --pets
```

The `--pets` command uses the optional cached 64-pixel Pets training images; if
the cache is absent it prints a skip message. Compare with
`from_scratch\warmup\pca_svd.py` only after the test passes or you are fully
stuck.

### Run the three plots

Run from the repository root. All three accept `--out` and `--show`; PCA also
accepts `--pets`. Each saves its PNGs under `from_scratch\runs\warmup\` by
default.

```powershell
python from_scratch\warmup\perceptron.py --out from_scratch\runs\warmup
python from_scratch\warmup\knn.py --out from_scratch\runs\warmup
python from_scratch\warmup\pca_svd.py --out from_scratch\runs\warmup
python from_scratch\warmup\pca_svd.py --out from_scratch\runs\warmup --pets
```

Recorded VM output (the metric text is exact; each line also prints the PNG
path):

```text
perceptron mistakes=0 after 2 epochs
best k=5, test accuracy=1.000
PC1 SVD/power absolute cosine=1.000000
```

The three scripts took about 1.63 s, 1.75 s, and 3.06 s, respectively, on the
2-vCPU VM. The `--pets` run also wrote `eigenpets.png` and
`pets_reconstructions.png`. These are measured VM results, not a guarantee for
every Windows laptop.

## 4. Download the Pets data

Build a NumPy/Pillow data pipeline that downloads the official Oxford-IIIT
Pets image and annotation archives, makes a train/validation split from the
official trainval list, and caches resized RGB images and trimaps. The download
is about **810 MB** compressed. On the VM, the populated
`from_scratch\data\` directory occupied **1.7 GB** (`du -sh`); archives,
extracted files, and caches account for the larger disk footprint.

The trimap labels are 1=foreground, 2=background, and 3=border/foreground in
this project's binary target (`mask = trimap != 2`). Images are resized
bilinearly, masks with nearest-neighbor interpolation, then images are
ImageNet-normalized when loaded. During training, horizontally flip both
arrays together. Sample either a jittered box or up to three point prompts:
first point positive, later points positive/negative, labels 1/0, box corners
2/3, and unused slots −1. A jittered box moves each edge by up to 10% of the
corresponding box side.

Hints:

<details><summary>Hint 1</summary> Keep raw downloads separate from cached arrays. Use the official `trainval.txt` and `test.txt` rather than guessing class membership from filenames. </details>

<details><summary>Hint 2</summary> Open every image with Pillow and convert to RGB before resizing. Use nearest neighbor for the trimap so interpolation cannot invent class IDs. </details>

<details><summary>Hint 3</summary> Sample foreground coordinates with `np.where(mask)` and background coordinates with `np.where(~mask)`. Skip masks that become all foreground or all background at the target size; they cannot supply both prompt classes. </details>

Download/build the 64-pixel cache and run the synthetic prompt tests:

```powershell
python from_scratch\scripts\download_data.py --size 64
py -3.11 -m pytest from_scratch\tests\test_data.py -q
```

The tests prove point labels/padding, box corner labels, and seeded sampling.
The reference files are `from_scratch\tinysam\data.py` and
`from_scratch\tests\test_data.py`.

## 5. Build the NumPy TinySAM, one file at a time

Use the dependency order below. Each checkpoint has a test command. The
`test_every_numpy_layer_input_and_parameter_gradients` test uses centered
finite differences on float64 inputs and sampled parameter entries. The
full-model check uses a tiny 16-pixel config so it is practical on CPU.

### 5.1 Linear

Build an affine map for arbitrary leading dimensions:
`Y=XW+b`. Its backward is `dX=dYWᵀ`, `dW=ΣXᵀdY` and `db=ΣdY`, summing over
batch/sequence axes but not the feature axis. See
[README: Linear and normalization](README.md#linear-and-normalization).

Hints:

<details><summary>Hint 1</summary> Store `W` as `(input_features, output_features)` so the forward is `X @ W`. </details>

<details><summary>Hint 2</summary> Flatten only the leading positions when computing `dW`; reshape `dX` back to the original input shape. </details>

<details><summary>Hint 3</summary> Accumulate gradients with `+=` because several paths may contribute to the same parameter before the optimizer step. </details>

Test and reference:

```powershell
py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py `
  -k test_every_numpy_layer_input_and_parameter_gradients -q
```

After it passes or you are fully stuck, compare with
`from_scratch\tinysam\numpy_impl\layers.py`.

### 5.2 LayerNorm

Normalize over the last feature axis using population variance:
`x̂=(x−μ)/sqrt(σ²+ε)`, `y=γx̂+β`. For `g=dY⊙γ`,
`dX=inv_std·(g−mean(g)−x̂·mean(g⊙x̂))`; reduce `dγ` and `dβ` over all
leading axes. See [README: Linear and normalization](README.md#linear-and-normalization).

Hints:

<details><summary>Hint 1</summary> Keep the mean and inverse standard deviation dimensions with `keepdims=True`; this makes broadcasting back over the feature vector clear. </details>

<details><summary>Hint 2</summary> Normalize with the population mean of squared centered values, not the sample-variance correction. </details>

<details><summary>Hint 3</summary> Compute `dγ=Σ(dY⊙x̂)` and `dβ=ΣdY` over every axis except the final one, then compare the input derivative to centered finite differences. </details>

Test: `py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_every_numpy_layer_input_and_parameter_gradients -q`.
Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\layers.py`.

### 5.3 GELU and ReLU

Build ReLU `max(0,x)` with derivative `1[x>0]`. TinySAM's GELU uses the
tanh approximation
`0.5x(1+tanh(c(x+0.044715x³)))`, `c=sqrt(2/π)`. Its derivative is given in
[README: GELU, ReLU, and softmax](README.md#gelu-relu-and-softmax).

Hints:

<details><summary>Hint 1</summary> Cache the input for GELU and the positive mask for ReLU during the forward pass; do not approximate GELU with a different library function. </details>

<details><summary>Hint 2</summary> For GELU, differentiate both the leading `x` and the `tanh(u(x))` factor. </details>

<details><summary>Hint 3</summary> Use `1−tanh(u)²` for the derivative of `tanh(u)` and multiply by `du/dx=c(1+3·0.044715x²)`. </details>

Test: `py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_every_numpy_layer_input_and_parameter_gradients -q`.
Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\layers.py`.

### 5.4 Stable softmax

For each row, calculate `p=exp(x−max(x))/Σexp(x−max(x))`. Its Jacobian-vector
product is `dx=p⊙(dp−Σ(dp⊙p))`. See
[README: GELU, ReLU, and softmax](README.md#gelu-relu-and-softmax).

Hints:

<details><summary>Hint 1</summary> Subtract the maximum along the softmax axis before exponentiating. </details>

<details><summary>Hint 2</summary> Do not build the full Jacobian. The backward needs one reduction `Σ(dp⊙p)` per softmax row. </details>

<details><summary>Hint 3</summary> Retain dimensions in the reduction so the scalar row correction broadcasts over classes. </details>

Test: `py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_attention_and_softmax_gradients -q`.
Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\layers.py`.

### 5.5 Multi-head attention

For one head, build `Q=XWq`, `K=XWk`, `V=XWv`,
`S=QKᵀ/sqrt(dh)`, `P=softmax(S)`, and `O=PV`, then merge heads and apply the
output projection. The backward uses `dV=PᵀdO`, `dP=dOVᵀ`, the softmax
Jacobian-vector product above, and `dQ=dSK/sqrt(dh)`,
`dK=dSᵀQ/sqrt(dh)`. See
[README: Attention and transformer blocks](README.md#attention-and-transformer-blocks).

Hints:

<details><summary>Hint 1</summary> Keep the batch and head axes separate: a common internal shape is `(batch, heads, sequence, head_width)`. </details>

<details><summary>Hint 2</summary> Support self-attention (`q=k=v`) and cross-attention (`q`, `k`, `v` with different sequence lengths). </details>

<details><summary>Hint 3</summary> After backpropagating through the output projection, split the context gradient into heads before calculating `dP`, `dV`, and `dS`. </details>

Test: `py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_attention_and_softmax_gradients -q`.
Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\layers.py`.

### 5.6 MLP

Build `Linear → GELU → Linear` for the image encoder and
`Linear → ReLU → Linear → ReLU → Linear` for the decoder heads. Backpropagate
in reverse order using the chain rule. The test covers both the two-layer
`MLP` and three-layer `MLP3`.

Hints:

<details><summary>Hint 1</summary> Let each child module cache its own forward values; the MLP only needs to compose their forward and backward calls. </details>

<details><summary>Hint 2</summary> The incoming gradient first passes through the last Linear, then the activation, then the first Linear. </details>

<details><summary>Hint 3</summary> Parameter names should remain dotted and stable (for example, `fc1.weight`) so weight conversion and the full-model tests can find them. </details>

Test: `py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_every_numpy_layer_input_and_parameter_gradients -q`.
Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\layers.py`.

### 5.7 Patch embedding

Split an image into non-overlapping `P×P` RGB patches, flatten each to
`3P²`, then apply a Linear map into `D` features. This is equivalent to a
convolution with kernel and stride `P`. Backward applies the Linear derivative,
then reverses the reshape/transposition.

Hints:

<details><summary>Hint 1</summary> For NCHW input, expose patch row, patch column, channel, and within-patch coordinates before flattening. </details>

<details><summary>Hint 2</summary> Do not overlap or interpolate patches; the output token count is `(S/P)²`. </details>

<details><summary>Hint 3</summary> The final backward reshape must exactly invert the forward axes. Verify with an asymmetric image shape in your own small test before the square model. </details>

Test: `py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_every_numpy_layer_input_and_parameter_gradients -q`.
Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\layers.py`.

### 5.8 UpsampleConvT2x and LayerNorm2d

With a transposed convolution whose kernel equals stride 2, pixels do not
overlap: map each input vector from `Cin` to `4·Cout`, reshape those values
into `(Cout,2,2)`, and pixel-shuffle. Backward applies the inverse shuffle
before the Linear derivative. `LayerNorm2d` normalizes channels independently
at each spatial location. The equations are in
[README: 2× upscaler and hypernetworks](README.md#equations-and-reverse-mode-derivatives).

Hints:

<details><summary>Hint 1</summary> The transposed-convolution output height and width are each exactly doubled. There is no summation of overlapping kernel windows in this special case. </details>

<details><summary>Hint 2</summary> Reshape the projected channel axis into `(Cout,2,2)` and interleave those two spatial axes with height and width. </details>

<details><summary>Hint 3</summary> For `LayerNorm2d`, move or reduce over the channel axis only; keep each pixel's statistics independent. </details>

Test: `py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_every_numpy_layer_input_and_parameter_gradients -q`.
Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\layers.py`.

### 5.9 Finite-difference gradcheck utility

Build a centered numerical derivative
`(f(x+ε)−f(x−ε))/(2ε)` and symmetric relative error
`|analytic−numeric|/max(|analytic|+|numeric|,1e−8)`. Use float64 and test a
small random subset of large parameter arrays. The tests exercise this utility
on layers, attention, losses, and the full model.

Hints:

<details><summary>Hint 1</summary> Save the original scalar value, perturb one selected entry in each direction, evaluate the scalar objective, then restore the entry. </details>

<details><summary>Hint 2</summary> Compare gradients at the exact same indices. Pick inputs away from ReLU's nondifferentiable zero. </details>

<details><summary>Hint 3</summary> Choose `ε=1e-5`, calculate in float64, and seed the sampled indices so a failing entry can be reproduced. </details>

Tests that exercise it:

```powershell
py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py `
  -k test_every_numpy_layer_input_and_parameter_gradients -q
py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py `
  -k test_full_tinysam_total_loss_parameter_gradients -q
```

Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\gradcheck.py`.

### 5.10 ImageEncoder

Build patch embedding plus learned absolute position vectors, then pre-LayerNorm
residual ViT blocks
`x←x+Attention(LN(x)); x←x+MLP(LN(x))`. Finish with the two Linear/LayerNorm
neck maps. Preserve `(B,Hp²,E)` token layout; `Hp=S/P`.

Hints:

<details><summary>Hint 1</summary> Add the learned positional embedding after patch embedding. Initialize it from a small zero-mean normal distribution. </details>

<details><summary>Hint 2</summary> In each residual block, normalize the input to the sublayer, then add that sublayer's output to the unnormalized residual stream. </details>

<details><summary>Hint 3</summary> The neck changes `D` to `E`, normalizes, applies a second `E→E` map, and normalizes again. </details>

Test: `py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_full_tinysam_total_loss_parameter_gradients -q`.
This checks the encoder parameters as part of TinySAM. Reference after
passing/stuck: `from_scratch\tinysam\numpy_impl\model.py`.

### 5.11 PromptEncoder

For a pixel coordinate `(x,y)`, use the half-pixel offset, normalize by image
size, map to `u=2c−1`, then encode
`[sin(2πuG), cos(2πuG)]` using a fixed seeded Gaussian matrix `G`. Add the
learned label embedding for positive, negative, box-corner, or padding
prompts; padding has zero positional encoding. See
[README: Prompt Fourier features](README.md#prompt-fourier-features).

Hints:

<details><summary>Hint 1</summary> Keep Fourier frequencies fixed and untrained; the trainable table is the small set of prompt-label vectors. </details>

<details><summary>Hint 2</summary> Use labels 1/0 for positive/negative points, 2/3 for box corners, and −1 for padding. The public coordinates are `(x,y)`, not `(row,column)`. </details>

<details><summary>Hint 3</summary> Use the same encoding function for sparse prompts and patch-center dense positional encodings; make the padding branch explicit. </details>

Test: `py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_full_tinysam_total_loss_parameter_gradients -q`.
Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\model.py`.

### 5.12 TwoWayTransformer

Implement each SAM-style decoder layer in order: token self-attention,
token-to-image attention, a `E→2E→E` ReLU MLP, then image-to-token attention.
Each sublayer has a residual and post-LayerNorm. Add the final token-to-image
attention and normalization. See
[README: Attention and transformer blocks](README.md#attention-and-transformer-blocks).

Hints:

<details><summary>Hint 1</summary> The query positional encoding is the original output-token tensor; use it for query/key additions in the cross-attention stages. </details>

<details><summary>Hint 2</summary> In token-to-image attention, queries are tokens and keys/values are image embeddings. In image-to-token attention, image positions query the tokens. </details>

<details><summary>Hint 3</summary> Write one layer's forward and reverse pass in a diagram before composing multiple layers; reverse the sublayer order exactly in backward. </details>

Test: `py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_full_tinysam_total_loss_parameter_gradients -q`.
Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\model.py`.

### 5.13 MaskDecoder

Prepend a learned IoU token and `K` learned mask tokens to sparse prompts.
Decode them with the two-way transformer. Upscale image features by 4× using
two 2× pixel-shuffle stages. Each mask token drives a small hypernetwork that
produces channel weights; dot those with the upscaled feature map to get mask
logits. A separate MLP predicts `K` IoU scores.

Hints:

<details><summary>Hint 1</summary> The decoded IoU token and mask tokens have fixed positions before the variable sparse-prompt tokens. </details>

<details><summary>Hint 2</summary> The mask-logit operation is a channel dot product at every output pixel, not a separate convolution with a spatial kernel. </details>

<details><summary>Hint 3</summary> Return `(B,K,4Hp,4Hp)` logits and `(B,K)` IoU predictions; verify both shapes before debugging their gradients. </details>

Test: `py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_full_tinysam_total_loss_parameter_gradients -q`.
Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\model.py`.

### 5.14 TinySAM and its loss

Assemble image encoder, prompt encoder, and mask decoder. Expose a full
forward, `encode_image`, and `decode` so repeated prompts can reuse image
features. Downsample the target by area averaging and thresholding at 0.5.
For each mask, use `20·focal + dice`; select the lowest mask loss per sample.
Train all IoU outputs against the constant IoU target with MSE. The full
derivations are in [README: Losses and optimizer](README.md#losses-and-optimizer).

Hints:

<details><summary>Hint 1</summary> Keep image encoding independent of prompt decoding. The app caches the embedding for multiple clicks on one image. </details>

<details><summary>Hint 2</summary> Only the best-of-K mask receives mask-loss gradients for each sample; the IoU-head MSE still covers every mask score. </details>

<details><summary>Hint 3</summary> Use the exact low-resolution target shape required by the mask logits. Keep the IoU target detached/constant in the PyTorch version. </details>

Test the assembled model:

```powershell
py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py `
  -k test_full_tinysam_total_loss_parameter_gradients -q
```

Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\model.py` and
`from_scratch\tinysam\numpy_impl\losses.py`.

### 5.15 Loss derivatives

Implement focal loss, soft Dice, and the IoU MSE with analytic logit
derivatives. For `p=σ(z)`, focal loss uses the chosen `α` and `γ`; Dice is
`1−(2Σpy+ε)/(Σp+Σy+ε)`. Apply the sigmoid derivative
`dp/dz=p(1−p)` when differentiating Dice. See the full focal derivative,
Dice derivative, reduction conventions, and best-of-K rule in
[README: Losses and optimizer](README.md#losses-and-optimizer).

Hints:

<details><summary>Hint 1</summary> Derive the gradient with respect to probability or logit explicitly; do not reuse the ordinary BCE gradient for focal loss. </details>

<details><summary>Hint 2</summary> Keep the pixel and batch/mask reductions consistent between the scalar loss and its gradient. </details>

<details><summary>Hint 3</summary> The binary IoU target comes from thresholded masks and is treated as a constant; differentiate only the MSE with respect to the predicted IoU. </details>

Test:

```powershell
py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py `
  -k test_losses_and_analytic_gradients -q
```

Reference after passing/stuck: `from_scratch\tinysam\numpy_impl\losses.py`.

### 5.16 AdamW and the training script

Implement AdamW's first/second moments, bias correction, and decoupled weight
decay:
`m←β₁m+(1−β₁)g`, `v←β₂v+(1−β₂)g²`,
`θ←θ(1−lr·wd)−lr·m̂/(sqrt(v̂)+ε)`. Then connect data batches, prompt sampling,
forward/loss/backward, warmup plus cosine learning rate, validation, CSV, and
checkpoint output in `train_numpy.py`. The equations are in
[README: Losses and optimizer](README.md#losses-and-optimizer).

Hints:

<details><summary>Hint 1</summary> Use one pair of moment arrays per parameter and increment one shared step counter for bias correction. </details>

<details><summary>Hint 2</summary> Apply weight decay directly to the parameter, separately from the adaptive gradient update; do not add `wd*parameter` to the gradient. </details>

<details><summary>Hint 3</summary> Begin with one optimizer update and one training step. Only add epoch loops, validation subsets, and plots after that works. </details>

The reference `test_numpy_gradcheck.py` has **no AdamW-specific unit test**:
its full-model test checks the gradients fed to the optimizer, not the update
rule. Verify the optimizer and training integration with a one-step smoke run
(after building your own training script, point this command at it):

```powershell
python from_scratch\scripts\train_numpy.py --epochs 1 --max-steps 1 `
  --batch-size 4 --steps-per-epoch 1 --warmup-steps 1 --val-subset 1 `
  --name adamw-smoke --out-dir from_scratch\runs
```

For gradient verification use
`py -3.11 -m pytest from_scratch\tests\test_numpy_gradcheck.py -k test_full_tinysam_total_loss_parameter_gradients -q`.
After the smoke run passes or you are fully stuck, compare with
`from_scratch\tinysam\numpy_impl\optim.py` and
`from_scratch\scripts\train_numpy.py`.

> **Debugging a failing gradcheck**
>
> - Test one layer at a time before checking the whole model.
> - Use float64 throughout the analytical and numerical calculation.
> - Print the input, output, gradient, and parameter shapes; verify that
>   reshapes and transposes are exact inverses.
> - Check every reduction axis: Linear sums over batch/sequence positions;
>   LayerNorm reduces features; softmax reduces its probability axis.
> - Compare the same sampled entries on both sides. Avoid ReLU inputs exactly
>   equal to zero, where the derivative is not unique.
> - If one entry fails, inspect its central-difference values at `+ε` and
>   `−ε` before increasing `ε` or changing the implementation.

## 6. Train locally on CPU

Once your tests pass, start with 10 epochs:

```powershell
python from_scratch\scripts\train_numpy.py --epochs 10 --batch-size 32 `
  --name pets-numpy
```

The output is under `from_scratch\runs\pets-numpy\`: weights, `loss.csv`,
`val_metrics.csv`, and `loss_curve.png`. `--max-steps` is useful for a quick
smoke test; `--val-subset` controls validation cost. The full reference run
used 110 epochs, took about **57.5 minutes**, and finished with train loss
**0.46328** on a **2-vCPU VM**. A batch-32 NumPy step measured about **0.27 s**
on that VM. The student's i5-7440HQ has four CPU cores, but its speed has not
been measured; expect a different runtime and start small.

Evaluate deterministic point and box prompts:

```powershell
python from_scratch\scripts\evaluate.py --impl numpy `
  --weights from_scratch\runs\pets-numpy\weights.npz --prompt point
python from_scratch\scripts\evaluate.py --impl numpy `
  --weights from_scratch\runs\pets-numpy\weights.npz --prompt box
```

The reference 110-epoch checkpoint scored test mIoU **0.7460** for point
prompts and **0.7603** for box prompts across 3,662 test images. A 10-epoch
run is not expected to match those numbers. Each evaluation writes JSON beside
the weights.

Make a prediction grid:

```powershell
python from_scratch\scripts\visualize.py --impl numpy `
  --weights from_scratch\runs\pets-numpy\weights.npz `
  --output from_scratch\runs\pets-numpy\visualize.png --count 8
```

The PNG has image-plus-prompt, ground truth, and predicted mask columns.

## 7. Build and check the PyTorch version

The parallel implementation is in
`from_scratch\tinysam\torch_impl\`: `layers.py`, `model.py`, `losses.py`,
`optim.py`, `mae.py`, and `convert.py`. Match the NumPy parameter names and
forward equations so the conversion/parity test can compare them. Use
`torch.nn.Parameter` and basic tensor operations with autograd. **Do not use**
`nn.Linear`, `nn.LayerNorm`, `nn.MultiheadAttention`, `torch.optim`, or
framework softmax shortcuts; the point is to implement the math yourself.
On this laptop use CPU-only Torch and do not target the 930MX.

Hints:

<details><summary>Hint 1</summary> Start by recreating Linear and LayerNorm using parameter tensors and elementary operations. Compare their outputs with your NumPy versions. </details>

<details><summary>Hint 2</summary> Keep dotted state-dict names aligned with the NumPy model; the conversion helper relies on matching parameter identities. </details>

<details><summary>Hint 3</summary> When autograd gradients disagree with NumPy, compare one parameter and one layer at a time before debugging full-model training. </details>

Run the parity test (forward outputs and every parameter gradient):

```powershell
py -3.11 -m pytest from_scratch\tests\test_numpy_torch_parity.py `
  -k test_numpy_and_torch_forward_and_all_parameter_gradients -q
```

The MAE checkpoint should contain only encoder tensors. The transfer test
requires exact equality for every saved `image_encoder.*` tensor:

```powershell
py -3.11 -m pytest from_scratch\tests\test_mae_transfer.py `
  -k test_mae_encoder_state_transfers_exactly -q
```

The reference CPU smoke reduced MAE loss from **1.658** to **1.202** in 20
steps and TinySAM train loss from **3.994** to **2.270** in 50 steps. These
were smoke tests on a 2-vCPU VM, not quality claims.

## 8. Train the 256-pixel model in Google Colab

Use Colab for the larger `TORCH_SMALL` model. Open
`from_scratch/notebooks/train_colab.ipynb` from GitHub:

1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. Select the **GitHub** tab.
3. Search for `Mehdi-blip-stack/EfficientSAM`, choose branch
   `devin/1792273248-from-scratch-tinysam`, and open
   `from_scratch/notebooks/train_colab.ipynb`.
4. Choose **Runtime → Change runtime type → T4 GPU** when available.
   The notebook checks CUDA and uses automatic device selection.
5. Run cells from top to bottom:
   - check Torch/CUDA, clone this branch, and move into the project;
   - install `from_scratch/requirements.txt`;
   - download Pets and create a 256-pixel cache;
   - run the tests;
   - pretrain the encoder with masked pixel reconstruction;
   - train the promptable model;
   - evaluate point and box prompts, visualize predictions, and plot loss;
   - create and download `tinysam-results.zip`.

Free-tier session limits and disconnect behavior vary. Save or download
checkpoints often, and remember that a disconnected runtime may lose local
files. The notebook saves model weights and logs under
`runs/pets-torch\`. Colab GPU timing has **not** been measured for this
project; do not quote the VM timings as GPU performance.

The downloaded ZIP is named `tinysam-results.zip`. It contains paths beginning
with `runs/pets-torch/`. Extract it into the laptop's `from_scratch\` folder;
that creates the files in `from_scratch\runs\pets-torch\`. Do not extract
inside `runs\pets-torch\`, or you will create a duplicate nested
`runs\pets-torch\runs\pets-torch\` directory.

## 9. Run the demo locally

Start the CPU NumPy app from the repository root:

```powershell
python from_scratch\app\app.py --impl numpy `
  --weights from_scratch\app\weights\tinysam_numpy.npz `
  --server-name 127.0.0.1
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Upload an image, choose
**positive** or **negative**, and click to add prompts. Choose **box** and
click two opposite corners to define a box. **Clear prompts** resets the
selection. The app reuses the image embedding while you try more clicks.
Press `Ctrl+C` in the terminal to stop the server.

To use a PyTorch checkpoint downloaded from Colab:

```powershell
python from_scratch\app\app.py --impl torch `
  --weights from_scratch\runs\pets-torch\weights.pt `
  --server-name 127.0.0.1
```

This path is CPU-only on the described laptop and may be slower; the NumPy
checkpoint is the recommended demo there.

## 10. Presenting the project to your professor

- [ ] Show the model diagram and explain why this is a teaching-scale
      implementation, not a reproduction of SAM/EfficientSAM training.
- [ ] Run `py -3.11 -m pytest from_scratch\tests -q`; show finite-difference
      and NumPy/PyTorch parity results.
- [ ] Explain one forward equation and its backward derivative (Linear,
      LayerNorm, attention, or focal/Dice are good choices).
- [ ] Show the training loss curve and validation mIoU CSV/curve.
- [ ] Show the point/box test metrics and state which checkpoint and split
      produced them. The reference values are from a 2-vCPU VM.
- [ ] Open `visualize.png` and describe the prompt, ground truth, and
      prediction columns.
- [ ] Run the local Gradio app and demonstrate positive, negative, and box
      prompts.
- [ ] Be clear that Colab uses an ordinary pixel-reconstruction MAE
      stand-in, not SAMI feature distillation from a pretrained teacher.

## 11. Troubleshooting on Windows

| Problem | Try this |
|---|---|
| `py` is not found | Re-run the Python installer, enable **Add python.exe to PATH**, then open a new PowerShell window. `py -0p` lists discovered interpreters. |
| `Activate.ps1` is blocked | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell, then activate `.venv` again. |
| `ModuleNotFoundError` | `cd` to the repository root, activate `.venv`, choose `.venv\Scripts\python.exe` in VS Code, then run `py -3.11 -m pytest ...` rather than another Python installation. |
| PyTorch download is large | The CPU wheel is a large download. Keep the CPU index URL, wait for pip to finish, and check `python -c "import torch; print(torch.cuda.is_available())"`; `False` is correct here. |
| Gradio says the port is in use | Start with `--server-port 7861` and open `http://127.0.0.1:7861`. |
| Git/Python reports long paths | Clone to a short location such as `C:\src\EfficientSAM`, avoid deeply nested folders, and enable Windows long-path support if necessary. |

## 12. How to ask for help

Paste the **full error/traceback**, the exact command you ran, which guide step
you were on, and the Python version (`py --version`). Include the relevant
file/function name and the test command/output. Hints will come first so you
can keep doing the implementation yourself.
