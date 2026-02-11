# MojoBLAS #

## Description ##

MojoBLAS is a GPU-accelerated implementation of a basic linear algebra 
subprogram (BLAS) library in the programming language Mojo. This library is
an efficient, portable, and easy-to-use BLAS library built directly on Mojo’s native GPU
abstractions. It ensures the technical correctness of Level
1, 2, and 3 BLAS routines based on the original FORTRAN 77 specification, support
for many floating-point precision types in GPU kernels, multi-platform portability across
NVIDIA and AMD hardware, and matches performance in current C++ BLAS libraries.

## Installation Instructions ##

### Machines with NVIDIA / AMD GPUs ###

1. Set up your Pixi coding environment.
- If you don't have `pixi` already, you can install it with: 
    ```
    curl -fsSL https://pixi.sh/install.sh | sh
    ```
2. Restart the terminal
- Close and reopen the terminal.

- If `pixi` has been successfully installed, you will see the `pixi` version return with this test:
    ```
    pixi --version
    ```
3. That's it! To run the program, see the 'Usage' section.

### Macs ###

NOTE: Mojo currently has limitied support for Mac GPUs. Thus, we do not expect all tests to succeed on Mac machines. However, to provide full accessibility to our project, installation instructions have been included below. If you want full functionality of MojoBLAS, please consider using NVIDIA or AMD GPUs, which are fully supported by Mojo.

1. Set up your Pixi coding environment.
- If you don't have `pixi` already, you can install it with: 
    ```
    curl -fsSL https://pixi.sh/install.sh | sh
    ```
2. Ensure `pixi` paths are visible.
- Manually add `pixi` to PATH.
    - For zsh shell:
        ```
        echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.zshrc
        source ~/.zshrc
        ```
    - For bash shell:
        ```
        echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        ```
        - Sometimes this command fails to add the `pixi` path to the login shell, which means this path won't be remembered by future shells. If you are running into an issue where new bash shells can't find the pixi executable, try adding it to your bash profile with:
            ```
            echo 'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi' >> ~/.bash_profile
            source ~/.bash_profile
            ```
3. Check pixi for successful installation:
- If `pixi` has been successfully installed, you will see the `pixi` version return with this test:
    ```
    pixi --version
    ```
4. Ensure XCode is downloaded.
- Check your application folder with:
    ```
    ls /Applications/Xcode.app
    ```
- If you see a list of folder contents, you're good to go!
- If not, download XCode from the App Store, and follow these extra instructions:
    - Change the command line tools to point to XCode's toolchain with:
        ```
        sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
        ```
    - Accept XCode's license agreement:
        ```
        sudo xcodebuild -license accept
        ```
    - Download MetalToolchain:
        ```
        sudo xcodebuild -downloadComponent MetalToolchain
        ```
    - Check it was successfully downloaded with:
        ```
        xcrun --find metallib
        ```

## Usage ##

### Machines with NVIDIA / AMD GPUs ###

1. Run the test with:
    ```
    pixi run mojo test-level1.mojo
    ```

### Macs ###

1. Add arm64 support.
- First, add a workspace platfrom for osx-arm64 systesm by running:
    ```
    pixi workspace platform add osx-arm64
    ```
    - This will add a pixi.lock file for supported osx-arm64 machines.

2. Follow the instructions for NVIDIA / AMD GPUs to run the tests.


## Contact Information ##

For questions about this library, please contact the project leader [Tatiana Melnichenko](tdehoff@vols.utk.edu) 
or [Alexa Andershock](aandersh@vols.utk.edu).

## Acknowledgements ##

This project could not have been completed without contributions from 
Alexa Andershock, Gian Fernandez-Aleman, Tatiana Melnichenko, Jackson Mowry, 
Holden Roaten, and Jonah Weston.
