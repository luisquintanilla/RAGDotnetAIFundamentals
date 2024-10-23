# RAG Sample using .NET AI Fundamentals

This sample shows how to apply vanilla / simple RAG pattern using .NET fundamentals in the AI ecosystem such as:

- [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers/)
- [System.Numerics.Tensors](https://www.nuget.org/packages/System.Numerics.Tensors/)
- [Microsoft.Extensions.AI](https://www.nuget.org/packages/Microsoft.Extensions.AI/)
- [Microsoft.Extensions.VectorData.Abstractions](https://www.nuget.org/packages/Microsoft.Extensions.VectorData.Abstractions/)
- [ML.NET](https://www.nuget.org/packages/Microsoft.ML/)
- [ONNX Runtime](https://www.nuget.org/packages/Microsoft.ML.OnnxTransformer/)

Given a list of movies, this sample implements semantic search, a building block for RAG patterns. 

## Prerequisites

- [.NET 9 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/9.0)
- Visual Studio or Visual Studio Code

## Quick Start

1. Open in GitHub Codespaces

    [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/luisquintanilla/RAGDotnetAIFundamentals)

1. Download the [e5-small-v2 model](https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx?download=true) and save it to the *assets* directory. If you rename the file, make sure to update the model path used by the `generator` in *Program.cs*.

## Setup

1. Download the [e5-small-v2 model](https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx?download=true) and save it to the *assets* directory. If you rename the file, make sure to update the model path used by the `generator` in *Program.cs*.

## Run the application

1. Open the terminal and run the following command.

```csharp
dotnet run
```