#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

// # define bfloat16_t bfloat16_t_t

class KernelFusedQKVProjectTranspose {
public:
    __aicore__ inline KernelFusedQKVProjectTranspose() {}

    __aicore__ inline void Init(GM_ADDR hidden_states, 
        GM_ADDR w_q, GM_ADDR w_k, GM_ADDR w_v, 
        GM_ADDR b_q, GM_ADDR b_k, GM_ADDR b_v, 
        GM_ADDR q, GM_ADDR k, GM_ADDR v, 
        GM_ADDR workspace, const FusedQKVProjTransposeTilingData &tilingData) {
        this->tilingData = tilingData;

        hiddenGm_.SetGlobalBuffer((__gm__ bfloat16_t*)hidden_states, 0);
        wqGm_.SetGlobalBuffer((__gm__ bfloat16_t*)w_q, 0);
        wkGm_.SetGlobalBuffer((__gm__ bfloat16_t*)w_k, 0);
        wvGm_.SetGlobalBuffer((__gm__ bfloat16_t*)w_v, 0);
        bqGm_.SetGlobalBuffer((__gm__ bfloat16_t*)b_q, 0);
        bkGm_.SetGlobalBuffer((__gm__ bfloat16_t*)b_k, 0);
        bvGm_.SetGlobalBuffer((__gm__ bfloat16_t*)b_v, 0);
        qGm_.SetGlobalBuffer((__gm__ bfloat16_t*)q, 0);
        kGm_.SetGlobalBuffer((__gm__ bfloat16_t*)k, 0);
        vGm_.SetGlobalBuffer((__gm__ bfloat16_t*)v, 0);

        (void)workspace;
    }

    __aicore__ inline void Process() {
        // TODO: user kernel impl
        uint32_t B = tilingData.get_batch();
        uint32_t S = tilingData.get_seq_len();
        uint32_t H = tilingData.get_num_heads();
        uint32_t H_kv = tilingData.get_num_kv_heads();
        uint32_t D = tilingData.get_hidden();
        ASSERT(D % H == 0);
        ASSERT(D % H_kv == 0);
        uint32_t Dh = D / H;

        uint32_t tp = tilingData.get_tokens_per_block();

        uint32_t totalTokens = B * S;
        uint32_t totalBlocks = (totalTokens + tp - 1) / tp;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t tokenStart = blockIdx * tp;
        uint32_t tokenEnd = min(tokenStart + tp, totalTokens);

        if (tokenStart >= tokenEnd) {
            return;
        }

        __gm__ bfloat16_t* hidden_ptr = this->hiddenGm_.GetGlobalBuffer();
        __gm__ bfloat16_t* wq_ptr = this->wqGm_.GetGlobalBuffer(); // [D, D]
        __gm__ bfloat16_t* wk_ptr = this->wkGm_.GetGlobalBuffer();
        __gm__ bfloat16_t* wv_ptr = this->wvGm_.GetGlobalBuffer();
        __gm__ bfloat16_t* bq_ptr = this->bqGm_.GetGlobalBuffer(); // [D], no bias in Janus-pro
        __gm__ bfloat16_t* bk_ptr = this->bkGm_.GetGlobalBuffer();
        __gm__ bfloat16_t* bv_ptr = this->bvGm_.GetGlobalBuffer();
        __gm__ bfloat16_t* q_ptr = this->qGm_.GetGlobalBuffer(); // [B, H, S, Dh]
        __gm__ bfloat16_t* k_ptr = this->kGm_.GetGlobalBuffer();
        __gm__ bfloat16_t* v_ptr = this->vGm_.GetGlobalBuffer();
        
        bool has_bias = (bq_ptr != nullptr && bk_ptr != nullptr && bv_ptr != nullptr);

        // UB buffer
        LocalTensor<bfloat16_t> hidden_bf16 = LocalTensor<bfloat16_t>(D);
        LocalTensor<float> hidden_fp32 = LocalTensor<float>(D);

        // Q / K / V output
        LocalTensor<float> q_fp32 = LocalTensor<float>(D);
        LocalTensor<float> k_fp32 = LocalTensor<float>(D);
        LocalTensor<float> v_fp32 = LocalTensor<float>(D);

        LocalTensor<float> bq_fp32 = LocalTensor<float>(D);
        LocalTensor<float> bk_fp32 = LocalTensor<float>(D);
        LocalTensor<float> bv_fp32 = LocalTensor<float>(D);

        if (has_bias){
            for (uint32_t i = 0; i < D; ++i) {
                bq_fp32[i] = (float)bq_ptr[i];
                bk_fp32[i] = (float)bk_ptr[i];
                bv_fp32[i] = (float)bv_ptr[i];
            }
        } else {
            bq_fp32.SetValue(0.0f);
            bk_fp32.SetValue(0.0f);
            bv_fp32.SetValue(0.0f);
        }

        LocalTensor<bfloat16_t> tmp_bf16 = LocalTensor<bfloat16_t>(D);

        for (uint32_t i = tokenStart; i < tokenEnd; ++i) {
            uint32_t b = i / S;
            uint32_t s = i % S;
            
            // 1: GM -> UB hidden[b, s, :]
            __gm__ bfloat16_t* hidden_row_ptr = hidden_ptr + (b * S + s) * D;
            DataCopy(hidden_bf16, hidden_row_ptr, D);
            Cast(hidden_fp32, hidden_bf16, D);
            
            Matmul1xD_DxD(q_fp32, hidden_fp32, wq_ptr, D);
            Matmul1xD_DxD(k_fp32, hidden_fp32, wk_ptr, D);
            Matmul1xD_DxD(v_fp32, hidden_fp32, wv_ptr, D);

            for(uint32_t h = 0; h < H; ++h) {
                uint32_t base = h * Dh;
                for (uint32_t d = 0; d < Dh; ++d) {
                    uint32_t idx = base + d;
                    float val = q_fp32[idx];
                    tmp_bf16[d] = (bfloat16_t)val;
                }
                __gm__ bfloat16_t *q_dst = q_ptr + (b * H + h) * S * Dh + s * Dh;
                DataCopy(q_dst, tmp_bf16, Dh);

                for (uint32_t d = 0; d < Dh; ++d) {
                    uint32_t idx = base + d;
                    float val = k_fp32[idx];
                    tmp_bf16[d] = (bfloat16_t)val;
                }
                __gm__ bfloat16_t *k_dst = k_ptr + (b * H + h) * S * Dh + s * Dh;
                DataCopy(k_dst, tmp_bf16, Dh);

                for (uint32_t d = 0; d < Dh; ++d) {
                    uint32_t idx = base + d;
                    float val = v_fp32[idx];
                    tmp_bf16[d] = (bfloat16_t)val;
                }
                __gm__ bfloat16_t *v_dst = v_ptr + (b * H + h) * S * Dh + s * Dh;
                DataCopy(v_dst, tmp_bf16, Dh);
            }
        }
    }

    __aicore__ inline void Matmul1xD_DxD(LocalTensor<float> &y, 
        const LocalTensor<float> &x, 
        __gm__ bfloat16_t* w, uint32_t D) {
        
        for (uint32_t row = 0; row < D; ++row){
            float acc = 0.0f;
            __gm__ bfloat16_t* w_row = w + row * D;
            for (uint32_t col = 0; col < D; ++col){
                acc += x[col] * (float)w_row[col];
            }
            y[row] = acc;
        }
        
    }
private:
    FusedQKVProjTransposeTilingData tilingData;
    GlobalTensor<bfloat16_t> hiddenGm_, wqGm_, wkGm_, wvGm_, bqGm_, bkGm_, bvGm_;
    GlobalTensor<bfloat16_t> qGm_, kGm_, vGm_;
};


extern "C" __global__ __aicore__ 
    void fused_qkv_project_transpose(
        GM_ADDR hidden_states, 
        GM_ADDR w_q, GM_ADDR w_k, GM_ADDR w_v, 
        GM_ADDR b_q, GM_ADDR b_k, GM_ADDR b_v, 
        GM_ADDR q, GM_ADDR k, GM_ADDR v, 
        GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelFusedQKVProjectTranspose op;
    op.Init(hidden_states, w_q, w_k, w_v, b_q, b_k, b_v, q, k, v, workspace, tiling_data);
    op.Process();
}
