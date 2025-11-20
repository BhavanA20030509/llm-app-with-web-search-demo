# 🔍 LLM App with Web Search Demo Application

Demo LLM app with Web Search for the YouTube video.

Watch the video 👇

<a href="https://youtu.be/kNgx0AifVo0">
<img src="https://i.imgur.com/CRxAv3F.png" width="800">
</a>


## Prerequisites

- **🐍 Python3.11**
- [🦙 Ollama](https://ollama.com/download)
- Download LLM model to use:
  ```sh
  ollama pull llama3.2
  ```
- Download embedding model:

    ```sh
    ollama pull nomic-embed-text:latest
    ```


## 🔨 Application Setup

Create a virtual environment and install all dependencies.

```sh
make setup
```

Activate virtual environment and run:

```sh
playwright install
```


## 🚀 Run the application

```sh
make run
```


## 🛟 Getting Help

```sh
make

# OR

make help
```
