#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

#define BUFFER_NUM 2

class KernelFusedQKVProjectTranspose {
public:
    __aicore__ inline KernelFusedQKVProjectTranspose() {}

    __aicore__ inline void Init(GM_ADDR hidden_states, 
        GM_ADDR w_q, GM_ADDR w_k, GM_ADDR w_v, 
        GM_ADDR b_q, GM_ADDR b_k, GM_ADDR b_v, 
        GM_ADDR q, GM_ADDR k, GM_ADDR v, 
        GM_ADDR workspace,
        const TilingData& tilingData) {
        
        this->tilingData = tilingData;
        this->B = tilingData.batch;
        this->S = tilingData.seq_len;
        this->D = tilingData.hidden;
        this->nH = tilingData.num_heads;
        this->nHkv = tilingData.num_kv_heads;
        this->Dh = tilingData.head_dim;
        this->kvDh = tilingData.head_kv_dim;

        // PRINTF("[Kernel] Init begin, B: %d, S: %d, D: %d, nH: %d, Dh: %d.\n", B, S, D, nH, Dh);
        
        /*
            hiddenGm shape: [B, S, D]
            In custom GEMV, wqGm shape should be [ nH,(3*)Dh, D ]
        */
        this->hiddenGm.SetGlobalBuffer((__gm__ bfloat16_t*)hidden_states);
        this->wqGm.SetGlobalBuffer((__gm__ bfloat16_t*)w_q);
        this->wkGm.SetGlobalBuffer((__gm__ bfloat16_t*)w_k);
        this->wvGm.SetGlobalBuffer((__gm__ bfloat16_t*)w_v);
        this->bqGm.SetGlobalBuffer((__gm__ bfloat16_t*)b_q);
        this->bkGm.SetGlobalBuffer((__gm__ bfloat16_t*)b_k);
        this->bvGm.SetGlobalBuffer((__gm__ bfloat16_t*)b_v);
        this->qGm.SetGlobalBuffer((__gm__ bfloat16_t*)q);
        this->kGm.SetGlobalBuffer((__gm__ bfloat16_t*)k);
        this->vGm.SetGlobalBuffer((__gm__ bfloat16_t*)v);

        
        pipe.InitBuffer(xBuf, D * sizeof(bfloat16_t));
        pipe.InitBuffer(yBuf, 3 * Dh * sizeof(bfloat16_t));  
        pipe.InitBuffer(accBuf, 3 * Dh * sizeof(float)); 
        pipe.InitBuffer(xfBuf, D * sizeof(float));   

        pipe.InitBuffer(wfBuf, D * sizeof(float));
        pipe.InitBuffer(prodBuf, D * sizeof(float));
        pipe.InitBuffer(redWorkBuf, D * sizeof(float));

        pipe.InitBuffer(inQueueW, BUFFER_NUM, D * sizeof(bfloat16_t));    // w_row[D]
        pipe.InitBuffer(outQueueR, BUFFER_NUM, 1 * sizeof(float));       // reduce result (1 float)

    }

    __aicore__ inline void Process() {
        uint32_t head_id = GetBlockIdx();
        if (head_id >= nH) return;

        const uint32_t O = 3u * Dh;
        const uint64_t w_stride = (uint64_t)O * (uint64_t)D;
        const uint64_t head_base = w_stride * (uint64_t)head_id;

        LocalTensor<bfloat16_t> x = xBuf.AllocTensor<bfloat16_t>();      // [D]
        LocalTensor<bfloat16_t> y = yBuf.AllocTensor<bfloat16_t>();      // [O]
        LocalTensor<float>      acc = accBuf.AllocTensor<float>();       // [O]
        LocalTensor<float>      xfp32 = xfBuf.AllocTensor<float>();       // [D]

        DataCopy(x, hiddenGm[0], D);
        pipe_barrier(PIPE_ALL);
        Cast(xfp32, x, RoundMode::CAST_NONE, (int32_t)D);

        for (uint32_t o = 0; o < O; ++o) {
            acc.SetValue(o, 0.0f);
        }
        
        int32_t loopCount = (int32_t)O + (BUFFER_NUM - 1);
        for (int32_t i = 0; i < loopCount; ++i) {
            if (i < (int32_t)O) {
                CopyInW(head_base, (uint32_t)i);
            }
            if (i >= 1) {
                ComputeRow(xfp32);
                CopyOutAcc(acc, (uint32_t)(i - 1));
            }
        }
    
        Cast(y, acc, RoundMode::CAST_RINT, (int32_t)O);
    
        DataCopy(qGm[(uint64_t)Dh * head_id], y[0],        Dh);
        DataCopy(kGm[(uint64_t)Dh * head_id], y[Dh],       Dh);
        DataCopy(vGm[(uint64_t)Dh * head_id], y[2 * Dh],   Dh);
    
        // pipe_barrier(PIPE_ALL);
        xBuf.FreeTensor(x);
        yBuf.FreeTensor(y);
        accBuf.FreeTensor(acc);
        xfBuf.FreeTensor(xfp32);
    }

    __aicore__ inline void CopyInW(uint64_t head_base, uint32_t o) {
        LocalTensor<bfloat16_t> wLocal = inQueueW.AllocTensor<bfloat16_t>();
        DataCopy(wLocal, wqGm[head_base + (uint64_t)o * (uint64_t)D], D);
        inQueueW.EnQue(wLocal);
    }

    __aicore__ inline void ComputeRow(LocalTensor<float>& xfp32) {
        LocalTensor<bfloat16_t> wLocal = inQueueW.DeQue<bfloat16_t>();
    
        LocalTensor<float> wF = wfBuf.AllocTensor<float>();        // [D]
        LocalTensor<float> prod = prodBuf.AllocTensor<float>();    // [D]
        LocalTensor<float> work = redWorkBuf.AllocTensor<float>();    // [D]
        LocalTensor<float> rLocal = outQueueR.AllocTensor<float>(); // [1]
    
        Cast(wF, wLocal, RoundMode::CAST_NONE, (int32_t)D);
        Mul(prod, wF, xfp32, (int32_t)D);
        ReduceSum(rLocal, prod, work, (int32_t)D);
    
        outQueueR.EnQue(rLocal);

        inQueueW.FreeTensor(wLocal);
        redWorkBuf.FreeTensor(work);
        prodBuf.FreeTensor(prod);
        wfBuf.FreeTensor(wF);
    }

    __aicore__ inline void CopyOutAcc(LocalTensor<float>& acc, uint32_t o) {
        LocalTensor<float> rLocal = outQueueR.DeQue<float>();
        acc.SetValue(o, rLocal.GetValue(0));
        outQueueR.FreeTensor(rLocal);
    }
    
private:
    TilingData tilingData;
    TPipe pipe;
    
    GlobalTensor<bfloat16_t> hiddenGm, wqGm, wkGm, wvGm;
    GlobalTensor<bfloat16_t> bqGm, bkGm, bvGm;
    GlobalTensor<bfloat16_t> qGm, kGm, vGm;
    
    uint32_t B, S, D, nH, nHkv, Dh, kvDh;

    TBuf<TPosition::VECCALC> xBuf, yBuf, accBuf, xfBuf;
    TBuf<TPosition::VECCALC> wfBuf, prodBuf, redWorkBuf;

    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueW;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueR;
};

extern "C" __global__ __aicore__ 
    void fused_qkv_project_transpose(
        GM_ADDR hidden_states, 
        GM_ADDR w_q, GM_ADDR w_k, GM_ADDR w_v, 
        GM_ADDR b_q, GM_ADDR b_k, GM_ADDR b_v, 
        GM_ADDR q, GM_ADDR k, GM_ADDR v, 
        GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelFusedQKVProjectTranspose op;
    
    op.Init(hidden_states, w_q, w_k, w_v, b_q, b_k, b_v, q, k, v, workspace, tiling_data);
    op.Process();
}
