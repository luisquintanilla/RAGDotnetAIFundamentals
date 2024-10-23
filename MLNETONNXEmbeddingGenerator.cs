#pragma warning disable SYSLIB5001
using Microsoft.ML.Tokenizers;
using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using Microsoft.ML.Data;

public class MLNETOnnxEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly Tokenizer _tokenizer;
    private readonly string? _modelPath;

    public EmbeddingGeneratorMetadata Metadata {get;}

    public MLNETOnnxEmbeddingGenerator(Tokenizer tokenizer, string? modelPath = "")
    {
        _tokenizer = tokenizer;
        _modelPath = modelPath;
        Metadata = new EmbeddingGeneratorMetadata("MLNETOnnxEmbeddingGenerator");
    }    

    public void Dispose()
    {
        throw new NotImplementedException();
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        var input = Preprocess(_tokenizer, values);

        // Run inference
        var output = Infer(_tokenizer, input);

        // Apply post-processing operations
        var attentionMask = input.First().AttentionMask;
        var pooled = MeanPooling(output, attentionMask, new long[] { 1, attentionMask.Length, output.Length / attentionMask.Length });
        var normalized = NormalizeAndDivide(pooled, new long[] { 1, attentionMask.Length, output.Length / attentionMask.Length });

        // Return embeddings
        var embedding = new Embedding<float>(normalized);
        return Task.FromResult(new GeneratedEmbeddings<Embedding<float>>([embedding]));
    }

    public TService? GetService<TService>(object? key = null) where TService : class
    {
        throw new NotImplementedException();
    }

    private float[] Infer(Tokenizer tokenizer, IEnumerable<ModelInput> input)
    {
        var ctx = new MLContext();

        var dv = ctx.Data.LoadFromEnumerable(input);

        var pipeline = 
            ctx.Transforms.ApplyOnnxModel(_modelPath);

        var result = pipeline.Fit(dv).Transform(dv);

        var embeddings = result.GetColumn<float[]>("last_hidden_state").First();

        return embeddings;
    }

    private IEnumerable<ModelInput> Preprocess(Tokenizer tokenizer, IEnumerable<string> text)
    {
        // Tokenize text
        var tokens = tokenizer.EncodeToIds(text.ToString() ?? "");

        var input = new ModelInput{
            InputIds = tokens.Select(t => (long)t).ToArray(),
            AttentionMask = tokens.Select(t => 1L).ToArray(),
            TokenTypeIds = tokens.Select(t => 0L).ToArray()
        };        

        // Return input
        return new [] {input};
    }

    private float[] MeanPooling(float[] embeddings, long[] attentionMask, long[] shape)
    {
        //// Extract shapes
        var batchSize = (int)shape[0];
        var sequenceLength = (int)shape[1];
        var embeddingSize = (int)shape[2];

        // Create a tensor for attention mask
        var attentionMaskTensor = Tensor.ConvertSaturating<long, float>(Tensor.Create<long>(attentionMask, [batchSize, sequenceLength]));

        // Create a tensor for token embeddings
        var tokenEmbeddings = new ReadOnlyTensorSpan<float>(embeddings, [(nint)batchSize, (nint)sequenceLength, (nint)embeddingSize], []);

        // Add a dimension to attention mask [2,11,1]
        var unsqueezed = Tensor.Unsqueeze(attentionMaskTensor, 2);

        // Expand Attention [2,11,384]
        var expandedAttention = Tensor.Broadcast<float>(unsqueezed, tokenEmbeddings.Lengths);

        // Multiply unsqueezed tensor with token embeddings [2,11,384]
        // Implicit broadcasting
        var lhs = Tensor.Multiply<float>(unsqueezed, tokenEmbeddings);

        // Contains intermediate calculator of embedding and attention
        // Tensors summed across the first axis.
        // Results in tensor shapes [2,384]
        var numerator = Tensor.Create<float>([batchSize, embeddingSize]);
        var denominator = Tensor.Create<float>([batchSize, embeddingSize]);

        // Apply sums along first axis.
        for (var batch = 0; batch < batchSize; batch++)
        {
            var sumEmbedding = Tensor.Create<float>([1, embeddingSize]);
            var sumAttention = Tensor.Create<float>([1, embeddingSize]);
            for (var sequence = 0; sequence < sequenceLength; sequence++)
            {
                var embeddingSlice =
                    Tensor.Squeeze(lhs.Slice([batch..(batch + 1), sequence..(sequence + 1), 0..embeddingSize]));

                var attentionSlice =
                    Tensor.Squeeze(expandedAttention.Slice([batch..(batch + 1), sequence..(sequence + 1), 0..embeddingSize]));

                sumEmbedding = Tensor.Add<float>(sumEmbedding, embeddingSlice);
                sumAttention = Tensor.Add<float>(sumAttention, attentionSlice);
            }

            Tensor.SetSlice(numerator, sumEmbedding, [batch..(batch + 1), 0..embeddingSize]);
            Tensor.SetSlice(denominator, sumAttention, [batch..(batch + 1), 0..embeddingSize]);
        }

        // Divide numerator by denominator. Mean pooling.
        var result = Tensor.Divide<float>(numerator, denominator);

        // Return result
        return result.ToArray();
    }

    private float[] NormalizeAndDivide(float[] sentenceEmbeddings, long[] shape)
    {
        long batchSize = shape[0];
        int embeddingSize = (int)shape[2];

        // Create a tensor for the square of the embeddings
        var squaredEmbeddings = Tensor.Multiply<float>(sentenceEmbeddings, sentenceEmbeddings);

        // Create Tensor for sumSquaredEmbeddings
        var sumSquaredEmbeddings = Tensor.Create<float>([(nint)batchSize, 1]);

        // Sum the squared embeddings across the embedding dimension
        for (var batch = 0; batch < batchSize; batch++)
        {
            // Get the embeddings for the current batch
            var embeddings = squaredEmbeddings.Slice([0..embeddingSize]);

            // Sum the embeddings across the embedding dimension
            var clampedSumEmbedding = Math.Max(Tensor.Sum<float>(embeddings), 1e-9f);
            var sumEmbeddings = Tensor.Create<float>(new float[] { clampedSumEmbedding }, [1, 1]);

            // Set the sum of the squared embeddings for the current batch
            sumSquaredEmbeddings[(ReadOnlySpan<nint>)[batch, 0]] = sumEmbeddings[(ReadOnlySpan<nint>)[0, 0]];
        }

        // Calculate the square root of the sum of the squared embeddings
        var sqrtSumSquaredEmbeddings = Tensor.Sqrt<float>(sumSquaredEmbeddings);

        // Divide the sentence embeddings by the denominator
        var normalizedEmbeddings = Tensor.Divide<float>(sentenceEmbeddings, sqrtSumSquaredEmbeddings);

        // Return the normalized embeddings
        return normalizedEmbeddings.ToArray();
    }    
}