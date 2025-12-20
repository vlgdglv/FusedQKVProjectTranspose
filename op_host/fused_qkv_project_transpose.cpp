#include <iostream>
#include "fused_qkv_project_transpose_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        optiling::TilingData tiling;
        // std::cout << "[FQKV Tiling] Enter TilingFunc" << std::endl;
        // uint32_t coreNum = ascendcPlatform.GetCoreNumAic();
        // uint32_t vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
        // std::cout << "CoreNumAic: " << coreNum << ", CoreNumAiv" << vectorCoreNum << std::endl;

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
       
        int32_t num_heads = 32;
        int32_t num_kv_heads = 32;

        if (D % num_heads != 0) return  ge::GRAPH_FAILED;
        int32_t head_dim = D / num_heads;

        int32_t head_kv_dim = D / num_kv_heads; // should be the same in test cases

        uint32_t total_token = B * S;
        
        // std::cout << "[FQKV Tiling] B=" << B << " S=" << S << " D=" << D << " total tokens=" << total_token << std::endl;
        // std::cout << "[FQKV Tiling] num_heads=" << num_heads << " num_kv_heads=" << num_kv_heads << std::endl;
        // std::cout << "[FQKV Tiling] head_dim=" << head_dim << " head_kv_dim=" << head_kv_dim << std::endl;

        tiling.set_batch(B);
        tiling.set_seq_len(S);
        tiling.set_hidden(D);
        tiling.set_num_heads(num_heads);
        tiling.set_num_kv_heads(num_kv_heads);
        tiling.set_head_dim(head_dim);
        tiling.set_head_kv_dim(head_kv_dim);
        tiling.set_tokens_per_block(B * S);
        
        context->SetBlockDim(32);
   
        // Matmul tiling 
        matmul_tiling::MatmulApiTiling cubeTiling(ascendcPlatform);
        // matmul_tiling::MultiCoreMatmulTiling cubeTiling(ascendcPlatform); 
        // cubeTiling.SetDim(20);   
        cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
        cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
        cubeTiling.SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16);
        
        uint32_t orgM = total_token;   // B*S
        uint32_t orgN = head_dim * 3;      // head_dim = D/num_heads
        uint32_t orgK = D;
        cubeTiling.SetShape(orgM, orgN, orgK);
        cubeTiling.SetOrgShape(orgM, orgN, orgK);
        cubeTiling.SetBufferSpace(-1, -1, -1);
        cubeTiling.SetBias(false);
        if (cubeTiling.GetTiling(tiling.cube_tiling) == -1) {
            return ge::GRAPH_FAILED;
        }

        // for matmul
        uint64_t systemWorkspaceSize = static_cast<uint64_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
        // for fp32 bias
        uint64_t userWorkspaceSize = 0; //(uint64_t)num_heads * 3ull * (uint64_t)head_dim * sizeof(uint16_t);
        // std::cout << "[FQKV Tiling] systemWorkspaceSize=" << systemWorkspaceSize << " userWorkspaceSize=" << userWorkspaceSize << std::endl;
        
        size_t* workspaces = context->GetWorkspaceSizes(1);
        if (workspaces == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling.set_sys_workspace_size(systemWorkspaceSize);

        workspaces[0] =  systemWorkspaceSize + userWorkspaceSize;
        // workspaces[0] = 0;

        // save tiling data
        auto *raw = context->GetRawTilingData();
        tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
        raw->SetDataSize(tiling.GetDataSize());

        // std::cout << "[FQKV Tiling] SaveToBuffer done, GRAPH_SUCCESS" << std::endl;
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
