
#include "fused_qkv_project_transpose_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
      FusedQKVProjectTransposeTilingData tiling;
    
      // 0: hidden_states, shape [B, S, D]
      const gert::StorageShape* x_shape = context->GetInputShape(0);
      const gert::Shape& storage_shape = x_shape->GetStorageShape();
    
      if (storage_shape.GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
      }
    
      uint32_t B = static_cast<uint32_t>(storage_shape.GetDim(0));
      uint32_t S = static_cast<uint32_t>(storage_shape.GetDim(1));
      uint32_t D = 1;
      // 剩下所有维度乘起来当 hidden size（兼容 [B,S,D] / [B,S,*,*]）
      for (int i = 2; i < storage_shape.GetDimNum(); ++i) {
        D *= static_cast<uint32_t>(storage_shape.GetDim(i));
      }
    
      const gert::RuntimeAttrs *runtime_attrs = context->GetAttrs();
      if (runtime_attrs == nullptr) {
          return ge::GRAPH_FAILED;
      }

      const int64_t *num_heads_ptr = runtime_attrs->GetInt(0);
      const int64_t *num_kv_heads_ptr = runtime_attrs->GetInt(1);
      // const int64_t *head_dim_ptr = runtime_attrs->GetInt(2);

      int32_t num_heads    = (num_heads_ptr    != nullptr) ? *num_heads_ptr    : 0;
      int32_t num_kv_heads = (num_kv_heads_ptr != nullptr) ? *num_kv_heads_ptr : 0;
      // int32_t head_dim     = (head_dim_ptr     != nullptr) ? *head_dim_ptr     : 0;
      
      tiling.set_batch(B);
      tiling.set_seq_len(S);
      tiling.set_hidden(D);
      tiling.set_num_heads(num_heads);
      tiling.set_num_kv_heads(num_kv_heads);
    
      uint64_t total_tokens = static_cast<uint64_t>(B) * static_cast<uint64_t>(S);
      if (total_tokens == 0) {
        tiling.set_tokens_per_block(1);
        context->SetBlockDim(1);
      } else {
        // 一个很保守的 heuristic：
        // - token 少（decode S=1）时，每个 block 1 个 token；
        // - token 多一点时，每个 block x 个 token；
        uint32_t tokens_per_block = 1;
        if (total_tokens >= 32) {
          tokens_per_block = 8;
        }
    
        uint32_t block_dim = static_cast<uint32_t>(
            (total_tokens + tokens_per_block - 1) / tokens_per_block);
    
        if (block_dim == 0) {
          block_dim = 1;
        } else if (block_dim > 32) {
          block_dim = 32;
        }
    
        tiling.set_tokens_per_block(tokens_per_block);
        context->SetBlockDim(block_dim);
      }
      
      // save tiling data
      auto *raw = context->GetRawTilingData();
      tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
      raw->SetDataSize(tiling.GetDataSize());
    
      return ge::GRAPH_SUCCESS;
    }
}
    


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class FusedQKVProjectTranspose : public OpDef {
public:
    explicit FusedQKVProjectTranspose(const char* name) : OpDef(name)
    {
        this->Input("hidden_states")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("w_q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("w_k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("w_v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("b_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("b_k")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("b_v")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_heads").Int();
        this->Attr("num_kv_heads").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(FusedQKVProjectTranspose);
}
