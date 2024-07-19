# dioecy_classifier

### Installation - Windows 10+
1. Make sure you have at least `Python 3.10`
2. Create a virtual environment 
    <pre><code class="language-python">python -m venv .venv_dioecy</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
3. Activate the virtual environment
    <pre><code class="language-python">.\.venv_dioecy\Scripts\activate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
4. Run the setup script to install packages and initialize submodules (use Git Bash)
    <pre><code class="language-python">bash setup.sh</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

### Installation - Unix-based systems (Linux, macOS)
1. Make sure you have at least `Python 3.10`
2. Create a virtual environment 
    <pre><code class="language-python">python -m venv .venv_dioecy</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
3. Activate the virtual environment
    <pre><code class="language-python">source .venv_dioecy/bin/activate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
4. Run the setup script to install packages and initialize submodules
    <pre><code class="language-python">./setup.sh</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>


### Training
ResNet was trained using `train_resnet.py`. The `train_resnext.py` implementation was used to test ResNeXt, which is a slightly more complex architecture, but the results were comparable or worse. The `train.py` file was used to train a Swin-V2 classifier, but it didn't work well and kept predicting only a single class.

### Prediction
1. Install LeafMachine2 and use the `CensorArchivalComponents.py` and `CensorArchivalComponents.yaml` workflow to generate censored images.
2. Run the predict_resnet.py script.