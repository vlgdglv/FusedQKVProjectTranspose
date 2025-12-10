#include <iostream>
#include "fused_qkv_project_transpose_tiling.h"
#include "register/op_def_registry.h"
// #include "lib/matmul/matmul_tiling.h" 

// #include "lib/matmul_intf.h"
// using namespace AscendC;

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        std::cout << "[FQKV Tiling] Enter TilingFunc" << std::endl;

        optiling::TilingData tiling;

        
        // 0: hidden_states, shape [B, S, D]
        const gert::StorageShape* x_shape = context->GetInputShape(0);
        const gert::Shape& storage_shape = x_shape->GetStorageShape();

        // std::cout << "[FQKV Tiling] dim_num = " << storage_shape.GetDimNum() << std::endl;
        
        
        if (storage_shape.GetDimNum() < 2) {
            return ge::GRAPH_FAILED;
        }

        uint32_t B = static_cast<uint32_t>(storage_shape.GetDim(0));
        uint32_t S = static_cast<uint32_t>(storage_shape.GetDim(1));
        uint32_t D = 1;

        for (int i = 2; i < storage_shape.GetDimNum(); ++i) {
            D *= static_cast<uint32_t>(storage_shape.GetDim(i));
        }

        // std::cout << "[FQKV Tiling] B=" << B << " S=" << S << " D=" << D << std::endl;

        const gert::RuntimeAttrs *runtime_attrs = context->GetAttrs();
        if (runtime_attrs == nullptr) {
            return ge::GRAPH_FAILED;
        }
        
        // std::cout << "[FQKV Tiling] runtime_attrs->GetAttrNum()=" << runtime_attrs->GetAttrNum() << std::endl;
        // const int64_t *num_heads_ptr = runtime_attrs->GetInt(0);
        // const int64_t *num_kv_heads_ptr = runtime_attrs->GetInt(1);
        // const int64_t *head_dim_ptr = runtime_attrs->GetInt(2);

        // int32_t num_heads    = (num_heads_ptr    != nullptr) ? *num_heads_ptr    : 0;
        // int32_t num_kv_heads = (num_kv_heads_ptr != nullptr) ? *num_kv_heads_ptr : 0;
        int32_t num_heads = 32;
        int32_t num_kv_heads = 32;
        // std::cout << "[FQKV Tiling] num_heads=" << num_heads
            //   << " num_kv_heads=" << num_kv_heads << std::endl;

        tiling.set_batch(B);
        tiling.set_seq_len(S);
        tiling.set_hidden(D);
        tiling.set_num_heads(num_heads);
        tiling.set_num_kv_heads(num_kv_heads);

        std::cout << "[FQKV Tiling] Setting TilingData: B=" << B << " S=" << S << " D=" << D << " num_heads=" << num_heads << " num_kv_heads=" << num_kv_heads << std::endl;

        uint64_t total_tokens = static_cast<uint64_t>(B) * static_cast<uint64_t>(S);
        // std::cout << "[FQKV Tiling] total_tokens=" << total_tokens << std::endl;
        
        uint32_t tokens_per_block = 1;
        uint32_t block_dim = 1;
        if (total_tokens == 0) {
            tiling.set_tokens_per_block(1);
            context->SetBlockDim(1);
        } else {
            if (total_tokens >= 32) {
                tokens_per_block = 16;
            }
            block_dim = static_cast<uint32_t>((total_tokens + tokens_per_block - 1) / tokens_per_block);

            if (block_dim == 0) {
                block_dim = 1;
            } else if (block_dim > 32) {
                block_dim = 32;
            }

            tiling.set_tokens_per_block(tokens_per_block);
            context->SetBlockDim(block_dim);

            // std::cout << "[FQKV Tiling] tokens_per_block=" << tokens_per_block
                // << " block_dim=" << block_dim << std::endl;
        }

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
            
        if (false){
            // Matmul tiling 
            // matmul_tiling::MatmulApiTiling cubeTiling(ascendcPlatform);
            matmul_tiling::MultiCoreMatmulTiling cubeTiling(ascendcPlatform); 
            cubeTiling.SetDim(block_dim);   
            cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
            cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
            cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
            cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
            uint32_t M = tokens_per_block;
            uint32_t K = D;
            uint32_t N = D;
            cubeTiling.SetShape(M, N, K);
            cubeTiling.SetOrgShape(S, N, K);
            cubeTiling.SetBufferSpace(-1, -1, -1);
            cubeTiling.SetBias(true);

            if (cubeTiling.GetTiling(tiling.cube_tiling) == -1) {
                return ge::GRAPH_FAILED;
            }
        }
        
        // // for matmul
        // uint64_t systemWorkspaceSize = static_cast<uint64_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
        // // for fp32 bias
        // uint64_t userWorkspaceSize = static_cast<uint64_t>(3u) * static_cast<uint64_t>(D) * sizeof(float);

        // std::cout << "[FQKV Tiling] systemWorkspaceSize=" << systemWorkspaceSize << " userWorkspaceSize=" << userWorkspaceSize << std::endl;
        
        size_t* workspaces = context->GetWorkspaceSizes(1);
        // if (workspaces == nullptr) {
        //     return ge::GRAPH_FAILED;
        // }
        // workspaces[0] = 0;
        // tiling.set_sys_workspace_size(systemWorkspaceSize);

        (void)workspaces;
        // workspaces[0] = userWorkspaceSize + systemWorkspaceSize;

        // save tiling data
        auto *raw = context->GetRawTilingData();
        tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
        raw->SetDataSize(tiling.GetDataSize());


        std::cout << "[FQKV Tiling] SaveToBuffer done, GRAPH_SUCCESS" << std::endl;

        return ge::GRAPH_SUCCESS;
    }
}
    

namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        const gert::Shape* x_shape = context->GetInputShape(0);
    
        // q
        gert::Shape* q_shape = context->GetOutputShape(0);
        *q_shape = *x_shape;
    
        // k
        gert::Shape* k_shape = context->GetOutputShape(1);
        *k_shape = *x_shape;
    
        // v
        gert::Shape* v_shape = context->GetOutputShape(2);
        *v_shape = *x_shape;
    
        return GRAPH_SUCCESS;
    }
    
    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        context->SetOutputDataType(1, inputDataType);
        context->SetOutputDataType(2, inputDataType);
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
        // this->Attr("num_heads").Int();
        // this->Attr("num_kv_heads").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(FusedQKVProjectTranspose);
}
