using Microsoft.ML.Data;
using Microsoft.Extensions.VectorData;

public class ModelInput
{
    [ColumnName("input_ids")]
    public long[] InputIds { get; set; }

    [ColumnName("attention_mask")]
    public long[] AttentionMask { get; set; }

    [ColumnName("token_type_ids")]
    public long[] TokenTypeIds { get; set; }
}

public class ModelOutput
{
    [ColumnName("last_hidden_state")]
    public float[] LastHiddenState { get; set; }
}

public class Movie
{
    [VectorStoreRecordData] 
    public string Title {get;set;}

    [VectorStoreRecordData]
    public string Description {get;set;}
}

public class MovieEmbedding
{
    [VectorStoreRecordKey]
    public int Key {get;set;}

    [VectorStoreRecordData]
    public Movie Details {get;set;}

    [VectorStoreRecordVector(384)]
    public ReadOnlyMemory<float> Vector {get;set;}
}