#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

// # define bfloat16_t bfloat16_t_t

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
        B = tilingData.batch;
        S = tilingData.seq_len;
        D = tilingData.hidden;
        H = tilingData.num_heads;
        Hkv = tilingData.num_kv_heads;
        Dh = D / H;

        PRINTF("[Kernel] Init begin, B: %d, S: %d, D: %d, H: %d, Hkv: %d.\n", B, S, D, H, Hkv);
        
        uint64_t sysWsBytes = tilingData.sys_workspace_size;
        __gm__ char* ws_base = (__gm__ char*)workspace;
        __gm__ float* user_ws = (__gm__ float*)(ws_base + sysWsBytes);
        qbfp32Gm.SetGlobalBuffer(user_ws + 0 * D);
        kbfp32Gm.SetGlobalBuffer(user_ws + 1 * D);
        vbfp32Gm.SetGlobalBuffer(user_ws + 2 * D);

        uint32_t maxTokensPerBlock = tilingData.tokens_per_block;

        hiddenGm.SetGlobalBuffer((__gm__ bfloat16_t*)hidden_states);
        wqGm.SetGlobalBuffer((__gm__ bfloat16_t*)w_q);
        wkGm.SetGlobalBuffer((__gm__ bfloat16_t*)w_k);
        wvGm.SetGlobalBuffer((__gm__ bfloat16_t*)w_v);
        bqGm.SetGlobalBuffer((__gm__ bfloat16_t*)b_q);
        bkGm.SetGlobalBuffer((__gm__ bfloat16_t*)b_k);
        bvGm.SetGlobalBuffer((__gm__ bfloat16_t*)b_v);
        qGm.SetGlobalBuffer((__gm__ bfloat16_t*)q);
        kGm.SetGlobalBuffer((__gm__ bfloat16_t*)k);
        vGm.SetGlobalBuffer((__gm__ bfloat16_t*)v);
        
        pipe.InitBuffer(biasBufBf16, D * sizeof(bfloat16_t));   
        pipe.InitBuffer(biasBufFp32, D * sizeof(float));

        LocalTensor<bfloat16_t> bias_bf16 = biasBufBf16.AllocTensor<bfloat16_t>();
        LocalTensor<float> bias_fp32 = biasBufFp32.AllocTensor<float>();

        // DataCopy(bias_bf16, bqGm[0], D);
        // pipe_barrier(PIPE_ALL);
        // Cast(bias_fp32, bias_bf16, RoundMode::CAST_NONE, D);
        // pipe_barrier(PIPE_ALL);
        // DataCopy(qbfp32Gm[0], bias_fp32, D);
        // pipe_barrier(PIPE_ALL);

        // float q0 = bias_fp32.GetValue(0);
        // PRINTF("[Kernel] q-bias[0] fp32 = %f\n", q0);
        
        // // k
        // DataCopy(bias_bf16, bkGm[0], D);
        // pipe_barrier(PIPE_ALL);
        // Cast(bias_fp32, bias_bf16, RoundMode::CAST_NONE, D);
        // pipe_barrier(PIPE_ALL);
        // DataCopy(kbfp32Gm[0], bias_fp32, D);
        // pipe_barrier(PIPE_ALL);

        // float k0 = bias_fp32.GetValue(0);
        // PRINTF("[Kernel] k-bias[0] fp32 = %f\n", k0);

        // // v
        // DataCopy(bias_bf16, bvGm[0], D);
        // pipe_barrier(PIPE_ALL);
        // Cast(bias_fp32, bias_bf16, RoundMode::CAST_NONE, D);
        // pipe_barrier(PIPE_ALL);
        // DataCopy(vbfp32Gm[0], bias_fp32, D);
        // pipe_barrier(PIPE_ALL);
        
        // float v0 = bias_fp32.GetValue(0);
        // PRINTF("[Kernel] v-bias[0] fp32 = %f\n", v0);

        biasBufFp32.FreeTensor(bias_fp32);
        biasBufBf16.FreeTensor(bias_bf16);

        pipe.InitBuffer(inQueueX, D * sizeof(float));        
        pipe.InitBuffer(projBuf, maxTokensPerBlock * D * sizeof(float));

        // PRINTF("[Kernel] Register CubeTiling.\n");
        // REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.cube_tiling);
        // PRINTF("[Kernel] CubeTiling: M=%u, N=%u, Ka=%u, Kb=%u, hasBias=%u\n",
        //     tilingData.cube_tiling.M, 
        //     tilingData.cube_tiling.N,
        //     tilingData.cube_tiling.Ka,
        //     tilingData.cube_tiling.Kb,
        //     tilingData.cube_tiling.isBias);
         
        // REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr());
        // pipe.InitBuffer(outHeadFp32, Dh * sizeof(float));
        // pipe.InitBuffer(outHeadBf16, Dh * sizeof(bfloat16_t));
        
        // pipe.InitBuffer(inQueueW, D * sizeof(float));
        // pipe.InitBuffer(tmpCalc, D * sizeof(float));
        // pipe.InitBuffer(outQueueRow, D * sizeof(float));
        
        // pipe.InitBuffer(biasBufBf16, D * sizeof(bfloat16_t));
        // pipe.InitBuffer(biasBufFp32, D * sizeof(float));
        
        // pipe.InitBuffer(tmpCast, D * sizeof(bfloat16_t));
        // pipe.InitBuffer(reduceTmp, D * sizeof(float));
        // pipe.InitBuffer(reduceBuf, D * sizeof(float));

        
    }

    __aicore__ inline void Process() {
        // PRINTF("[Kernel] In Process.\n");
        
        uint32_t tokensPerBlock = tilingData.tokens_per_block;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tokenStart = blockIdx * tokensPerBlock;
        uint32_t tokenEnd = tokenStart + tokensPerBlock;
        uint32_t totalTokens = B * S;

        if (tokenStart >= tokenEnd) {
            return;
        }

        if (tokenEnd > totalTokens) {
            tokenEnd = totalTokens;
        }
        
        uint32_t M_tile = tokensPerBlock;
        if (tokenStart + M_tile > totalTokens) {
            M_tile = totalTokens - tokenStart;
        }
        
        PRINTF("[Kernel] blockIdx: %d, tokenStart: %d, tokenEnd: %d.\n", blockIdx, tokenStart, tokenEnd);
        
        // ComputeBlock(tokenStart, M_tile);

        // for (uint32_t i = tokenStart; i < tokenEnd; ++i) {
        //    ComputeToken(i);
        // }
        PRINTF("[Kernel] Do nothing.\n");
    }

    __aicore__ inline void ComputeBlock(uint32_t tokenStart, uint32_t M_tile) {
        
        auto gm_a = hiddenGm[tokenStart * D];
        auto gm_bq_bias = (GlobalTensor<float>&)bqGm;

        LocalTensor<float> proj_buf = projBuf.AllocTensor<float>();
            
        // Q
        mm.SetTensorA(gm_a);         // [M_tile, D]
        mm.SetTensorB(wqGm[0]);      // [D, D]
        mm.SetBias(qbfp32Gm[0]);   // [D]
        mm.IterateAll(proj_buf);       // A@W_q + bias -> proj_q
        mm.End();
        // SplitHeadAndStore(proj_q, tokenStart, M_tile, qGm);

        // K
        mm.SetTensorA(gm_a);         // [M_tile, D]
        mm.SetTensorB(wkGm[0]);      // [D, D]
        mm.SetBias(kbfp32Gm[0]);   // [D]
        mm.IterateAll(proj_buf);       // A@W_q + bias -> proj_q
        mm.End();
        // SplitHeadAndStore(proj_q, tokenStart, M_tile, qGm);

        mm.SetTensorA(gm_a);         // [M_tile, D]
        mm.SetTensorB(wvGm[0]);      // [D, D]
        mm.SetBias(vbfp32Gm[0]);   // [D]
        mm.IterateAll(proj_buf);       // A@W_q + bias -> proj_q
        mm.End();
        // SplitHeadAndStore(proj_q, tokenStart, M_tile, qGm);

        
        projBuf.FreeTensor(proj_buf);


    }

    // __aicore__ inline void ComputeToken(uint32_t tokenIdx) {
    //     LocalTensor<float> x_ub = inQueueX.AllocTensor<float>();
    //     LoadAndCast(x_ub, hiddenGm, tokenIdx * D, D);

    //     ComputeAndWrite(x_ub, wqGm, bqGm, qGm, tokenIdx);
    //     ComputeAndWrite(x_ub, wkGm, bkGm, kGm, tokenIdx);
    //     ComputeAndWrite(x_ub, wvGm, bvGm, vGm, tokenIdx);

    //     inQueueX.FreeTensor(x_ub);
    // }

    // __aicore__ inline void ComputeAndWrite(
    //     LocalTensor<float>& x_ub,
    //     GlobalTensor<bfloat16_t>& wGm,
    //     GlobalTensor<bfloat16_t>& bGm,
    //     GlobalTensor<bfloat16_t>& outGm,
    //     uint32_t tokenIdx
    // ) {
    //     LocalTensor<float> w_row_ub = inQueueW.AllocTensor<float>();
    //     LocalTensor<float> tmp_ub = tmpCalc.AllocTensor<float>();
    //     LocalTensor<float> res_ub = outQueueRow.AllocTensor<float>();

    //     LocalTensor<bfloat16_t> bias_bf16 = biasBufBf16.AllocTensor<bfloat16_t>();
    //     LocalTensor<float> bias_fp32 = biasBufFp32.AllocTensor<float>();
    //     LocalTensor<float> reduce_tmp = reduceTmp.AllocTensor<float>();
    //     LocalTensor<float> reduce_dst = reduceBuf.AllocTensor<float>();

    //     DataCopy(bias_bf16, bGm[0], D);
    //     pipe_barrier(PIPE_ALL);
        
    //     Cast(bias_fp32, bias_bf16, RoundMode::CAST_NONE, D);
    //     pipe_barrier(PIPE_ALL);

    //     for (uint32_t i = 0; i < D; ++i) {
    //         LoadAndCast(w_row_ub, wGm, i * D, D);

    //         Mul(tmp_ub, x_ub, w_row_ub, D);

    //         ReduceSum(reduce_dst, tmp_ub, reduce_tmp, D);

    //         float bias_val = bias_fp32.GetValue(i); 
    //         float dot_val = reduce_dst.GetValue(0);
    //         res_ub.SetValue(i, dot_val + bias_val);
    //     }

    //     biasBufBf16.FreeTensor(bias_bf16);
    //     biasBufFp32.FreeTensor(bias_fp32);

    //     LocalTensor<float> head_fp32 = outHeadFp32.AllocTensor<float>();
    //     LocalTensor<bfloat16_t> head_bf16 = outHeadBf16.AllocTensor<bfloat16_t>();
        
    //     uint32_t b = tokenIdx / S;
    //     uint32_t s = tokenIdx % S;

    //     for (uint32_t h = 0; h < H; ++h){
    //         for (uint32_t d = 0; d < Dh; ++d){
    //             float val = res_ub.GetValue(h * Dh + d);
    //             head_fp32.SetValue(d, val);
                
    //         }
    //         // PRINTF("[Kernel] before cast, res_ub:val=%f, head_fp32:val=%f\n", res_ub.GetValue(0), head_fp32.GetValue(0));

    //         Cast(head_bf16, head_fp32, RoundMode::CAST_RINT, Dh);
    //         pipe_barrier(PIPE_ALL);
            
    //         uint64_t gm_offset = (uint64_t)b * H * S * Dh + (uint64_t)h * S * Dh + (uint64_t)s * Dh;

    //         // PRINTF("[Kernel] b: %d, h: %d, s: %d, gm_offset: %d, val=%f\n", b, h, s, gm_offset, head_bf16.GetValue(0));

    //         DataCopy(outGm[gm_offset], head_bf16, Dh);
    //         pipe_barrier(PIPE_ALL);
    //     }
    //     inQueueW.FreeTensor(w_row_ub);
    //     tmpCalc.FreeTensor(tmp_ub);
    //     outQueueRow.FreeTensor(res_ub);
    //     reduceTmp.FreeTensor(reduce_tmp);

    //     outHeadFp32.FreeTensor(head_fp32);
    //     outHeadBf16.FreeTensor(head_bf16);
    // }

    // __aicore__ inline void LoadAndCast(
    //     LocalTensor<float> &dst,
    //     GlobalTensor<bfloat16_t> &src,
    //     uint32_t offset,
    //     uint32_t length
    // ){
    //     LocalTensor<bfloat16_t> tmp = tmpCast.AllocTensor<bfloat16_t>();
    //     DataCopy(tmp, src[offset], length);
    //     pipe_barrier(PIPE_ALL);
    //     Cast(dst, tmp, RoundMode::CAST_NONE, length);
    //     pipe_barrier(PIPE_ALL);
    //     tmpCast.FreeTensor(tmp);
    // }
        
    
private:
    TilingData tilingData;
 
    GlobalTensor<bfloat16_t> hiddenGm, wqGm, wkGm, wvGm;
    GlobalTensor<bfloat16_t> bqGm, bkGm, bvGm;
    GlobalTensor<bfloat16_t> qGm, kGm, vGm;
    GlobalTensor<float> qbfp32Gm, kbfp32Gm, vbfp32Gm;

    uint32_t B, S, D, H, Hkv, Dh;

    TPipe pipe;
    TBuf<TPosition::VECCALC> inQueueX;
    // TBuf<TPosition::VECCALC> outHeadFp32;
    // TBuf<TPosition::VECCALC> outHeadBf16;

    using aType     = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    using bType     = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
    using cType     = MatmulType<TPosition::VECIN, CubeFormat::ND, float>;
    using biasType  = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    Matmul<aType, bType, cType, biasType> mm;
    TBuf<TPosition::VECIN> projBuf;  

    TBuf<TPosition::VECCALC> biasBufBf16;
    TBuf<TPosition::VECCALC> biasBufFp32;


    // TBuf<TPosition::VECCALC> inQueueW;
    // TBuf<TPosition::VECCALC> tmpCalc;
    // TBuf<TPosition::VECCALC> outQueueRow;

    // TBuf<TPosition::VECCALC> tmpCast;
    // TBuf<TPosition::VECCALC> reduceTmp;
    // TBuf<TPosition::VECCALC> reduceBuf;
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
