# PyML Assignments

Welcome to PyML!!! :D 

I would hope that you all are excited to learn some practical machine learning. We're certainly excited to teach it. 

This will be the root directory for all of your assignments and where the virtual environment, and other global parameters reside. In future assignments you will only have the assignment itself without this root directory. 

## A Note on LLM Usage

**DO NOT USE LLMs TO WRITE CODE!**

A large part of this course is learning the nuances of Python, NumPy, and PyTorch. The more modalities you work with while learning, in this case typing and reading, and the more time you spend pondering and struggling through problems and bugs, the better you'll learn. Struggling through problems is an emotional feat, as well as intellectual, and emotion is very much required for learning. You lose a modality and you lose a lot of emotional content (aside from perhaps boredom) if you let an LLM do the homework from the start. 

With this, LLMs are very good at explaining things, and if you don't understand a particular concept I can suggest using a modern high-parameter model as a resource (ChatGPT, Claude, Gemini, etc.). Nothing in this course is particularly novel or niche, so the chances of hallucinations are pretty slim.

Further, learning requires a very particular level of difficulty. Things that are too easy are forgettable, boring, and generally have low emotional engagement. Things that are too hard might have high emotional engagement, but usually it's of a negative kind that leads to burnout and other things that no makes someone no longer want to engage with a topic. Ideally it's somewhere in the middle. Enough challenge to where you're emotionally engaged and get a timely sense of accomplishment, but not so easy that there was no struggle. With this, if after you've thought over a problem for a while and are left stuck, I would encourage asking an LLM for a hint, of even if you're really not following, for the solution. If you ask for the solution though, make sure that you fully understand it and redo the question on your own.


## Installation and Environment

### General Environment

This course expects that you work in a unix-like environment. If you're running Mac of some version of Linux you're already there. If you're running Windows, install the most recent Windows Subsystem for Linux (WSL). You can find a link with instructions below: 

- https://learn.microsoft.com/en-us/windows/wsl/about

Further, this course assumes basic familiarity with basic shell commands (`cd`, `ls`, `rm`, etc.). If you're not familiar with this, there are plenty of cheat sheets and simple tutorials online. 

### IDE

This course does not assume a particular IDE. Use whatever you're comfortable with. Personally, I like Neo Vim and VSCode. 

### `git`

We will be using git in this course. Make sure it is installed on your computer, for Windows users, this should be on your subsystem. Do this before installing `uv.` Instructions for installing `git` are found here:

- https://git-scm.com/install/

### `make`
`make` is a build automation tool that lets you define and run common commands through simple shortcuts. We'll be using it throughout the course, so make sure it's installed — though it almost certainly already is on Mac and Linux. Windows users should have it available on WSL as well.

### `uv`
`uv` is a fast Python package and project manager that we'll be using to manage our virtual environment and dependencies. Think of it as a modern, much speedier replacement for `pip` + `venv`. You can find installation instructions here:

- https://docs.astral.sh/uv/getting-started/installation

Windows users should install it on WSL. You'll almost never have to deal with `uv` directly in this course — just know that the commands we run are prefixed by `uv run`, or you can activate the environment directly with `source .venv/bin/activate` or `make activate` from the root homework directory.


## Checking Your Work and Understanding

### Checking Your Work

The primary way for you to check your work is through the course website in the homework section. 

- https://py.ml.tu-berlin.de/quiz/#/homework

Click on the assignment you're working on and you'll find a box that asks you to drop your `submission.json` file there. 

Feel free to submit often. There is no limit to the number of times that you can submit. Submitting will provide information about where your code does and does not work and will hopefully help you fix any bugs or errors in the process. This system was deliberately designed to help you. So please **SUBMIT EARLY AND OFTEN**.

You create your `submission.json` file using the `submit.py` program in the assignment's root directory. `make submit-all` or `python submit.py --all` will create a submission for all of your code.

 Run `make` to see the usage for specific shortcuts to this in the assignment's `Makefile` or run `python `

### Automated tests

Bundled with each assignment is the boilerplate code for unit tests in `pytest`. Please note: there are no actual tests in this code. It's there to help you consider certain aspects of your code and cases that your code will likely encounter. You should gain a sense of these cases from the test names themselves and any included comments. Feel free to fill out these unit tests as you so choose, particularly it the feedback from submitting isn't helpful. Unit tests are a valuable tool for working through difficult problems.

### Quizzes

There are some things that we can't test with code. Instead, we'll test with quizzes, all of which are on the course website (NOT ISIS). Look at the `README.md` for a given assignment to see if these are included.


## Using `git`

`git` is a version control system that tracks changes to your code over time. It lets you save snapshots of your work, experiment freely without fear of losing progress, and maintain a clean history of what you've done and why. It's an indispensable tool in any real engineering workflow, so we'll be using it throughout this course.

### `git init`

This initializes a new git repository in the current directory. Every assignment is its own repository, so you'll run this once at the root of each assignment when you start.

### `git branch`

A branch is essentially a parallel version of your repository — it lets you diverge from your main line of work without affecting it. You can list your branches by running `git branch`, and create a new one with `git branch <branch-name>`. It's good practice to create a new branch when you're working on a distinct aspect of an assignment, or if you want to try something experimental and don't want to risk messing up the progress you've already made.

### `git checkout`

`git checkout <branch-name>` switches you over to an existing branch. If you want to create a new branch and immediately switch to it in one step, you can use `git checkout -b <branch-name>` — you'll find yourself using this one a lot.

### `git commit`

A commit is a saved snapshot of your work at a particular point in time. Good moments to commit are when you've finished a logical chunk of work — implemented a function, fixed a bug, got a test passing — basically any time you've reached a small but meaningful milestone you wouldn't want to lose. When you run `git commit`, git will open a text editor for you to write a commit message. If you want to skip that and write the message inline, you can use the shorthand `git commit -m "Your message here"`.

A good commit message is short, specific, and written in the imperative mood — think "Add gradient descent implementation" rather than "stuff" or "added some things". The goal is that anyone (including future you) can skim the commit history and understand what changed and why without having to dig into the code.

### `git merge`

Merging is how you bring the work from one branch back into another — typically once you're happy with what you've done on a feature or experimental branch and want to fold it back into your main branch. To do this, switch to the branch you want to merge *into* with `git checkout <target-branch>`, then run `git merge <branch-to-merge>`. If the two branches have conflicting changes, git will flag them and ask you to resolve them manually before the merge completes.
