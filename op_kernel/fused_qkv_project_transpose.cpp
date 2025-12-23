#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
using namespace AscendC;


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
        pipe.InitBuffer(accBuf, 3 * Dh * sizeof(float));
        pipe.InitBuffer(yBuf, 3 * Dh * sizeof(bfloat16_t));   
        pipe.InitBuffer(wBuf, D * sizeof(bfloat16_t));   
        pipe.InitBuffer(xfBuf, D * sizeof(float));   
        pipe.InitBuffer(wfBuf, D * sizeof(float));
        pipe.InitBuffer(tmpBuf, 3 * D * sizeof(float));

        pipe.InitBuffer(prodBuf, D * sizeof(float));
        pipe.InitBuffer(redWorkBuf, D * sizeof(float));
        pipe.InitBuffer(redOutBuf, sizeof(float));

    }

    __aicore__ inline void Process() {
        uint32_t head_id = GetBlockIdx();
        if (head_id >= nH) return;

        const uint32_t O = 3u * Dh;
        const uint64_t w_stride = (uint64_t)O * (uint64_t)D;
    
        LocalTensor<bfloat16_t> x = xBuf.AllocTensor<bfloat16_t>();      // [D]
        LocalTensor<float>      acc = accBuf.AllocTensor<float>();       // [O]
        LocalTensor<float>      xfp32 = xfBuf.AllocTensor<float>();       // [D]
        LocalTensor<float>      wfp32 = wfBuf.AllocTensor<float>();       // [D]
        
        LocalTensor<bfloat16_t> y = yBuf.AllocTensor<bfloat16_t>();      // [O]
        LocalTensor<bfloat16_t> w = wBuf.AllocTensor<bfloat16_t>();      // [D]
        LocalTensor<float>      tmp = tmpBuf.AllocTensor<float>();       // [D]

        LocalTensor<float>      prod = prodBuf.AllocTensor<float>();     // [D]
        LocalTensor<float>      work = redWorkBuf.AllocTensor<float>();  // [D] (workspace)
        LocalTensor<float>      red  = redOutBuf.AllocTensor<float>();   // [1] (或 [D] 看你的 ReduceSum 需求)

    
        DataCopy(x, hiddenGm[0], D);
        pipe_barrier(PIPE_ALL);
        Cast(xfp32, x, RoundMode::CAST_NONE, (int32_t)D);

        for (uint32_t o = 0; o < O; ++o) {
            acc.SetValue(o, 0.0f);
        }
        
        for (uint32_t o = 0; o < O; ++o) {
            DataCopy(w, wqGm[w_stride * (uint64_t)head_id + (uint64_t)o * (uint64_t)D], D);
            pipe_barrier(PIPE_ALL);
            Cast(wfp32, w, RoundMode::CAST_NONE, (int32_t)D);
    
            Mul(prod, wfp32, xfp32, (int32_t)D);
            ReduceSum(red, prod, work, (int32_t)D);
            acc.SetValue(o, red.GetValue(0));
        }
    
        pipe_barrier(PIPE_ALL);
        Cast(y, acc, RoundMode::CAST_RINT, (int32_t)O);
        pipe_barrier(PIPE_ALL);
    
        DataCopy(qGm[(uint64_t)Dh * head_id], y[0],        Dh);
        DataCopy(kGm[(uint64_t)Dh * head_id], y[Dh],       Dh);
        DataCopy(vGm[(uint64_t)Dh * head_id], y[2 * Dh],   Dh);
    
        pipe_barrier(PIPE_ALL);
    
        yBuf.FreeTensor(y);
        accBuf.FreeTensor(acc);
        xBuf.FreeTensor(x);
        wBuf.FreeTensor(w);
    }


public:
    // using aType     = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    using aType     = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    using bType     = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    using cType     = MatmulType<TPosition::VECIN, CubeFormat::ND, bfloat16_t>;
    using biasType  = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    Matmul<aType, bType, cType, biasType> mm;
    // Matmul<aType, bType, cType, biasType> mmQ;
    // Matmul<aType, bType, cType, biasType> mmK;
    // Matmul<aType, bType, cType, biasType> mmV;
    TPipe pipe;

private:
    TilingData tilingData;

    GlobalTensor<bfloat16_t> hiddenGm, wqGm, wkGm, wvGm;
    GlobalTensor<bfloat16_t> bqGm, bkGm, bvGm;
    GlobalTensor<bfloat16_t> qGm, kGm, vGm;
    
    uint32_t B, S, D, nH, nHkv, Dh, kvDh;

    // TBuf<TPosition::VECOUT> aBuf;
    TBuf<TPosition::VECIN> tmpBuf;
    TBuf<TPosition::VECIN> cBuf;
    TBuf<TPosition::VECCALC> xBuf, yBuf, accBuf, wBuf;
    TBuf<TPosition::VECCALC> xfBuf, wfBuf, tmpfBuf;
    TBuf<TPosition::VECCALC> prodBuf;
    TBuf<TPosition::VECCALC> redWorkBuf;
    TBuf<TPosition::VECCALC> redOutBuf;


    // GlobalTensor<bfloat16_t> workGm;

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
    
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &tiling_data.cube_tiling);
    op.Init(hidden_states, w_q, w_k, w_v, b_q, b_k, b_v, q, k, v, workspace, tiling_data);
    op.Process();

}
