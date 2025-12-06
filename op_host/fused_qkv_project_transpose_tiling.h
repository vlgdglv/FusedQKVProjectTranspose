#include "register/tilingdata_base.h"

namespace optiling {
  BEGIN_TILING_DATA_DEF(FusedQKVProjectTransposeTilingData)
    // hidden_states: [B, S, D]
    TILING_DATA_FIELD_DEF(uint32_t, batch);            // B
    TILING_DATA_FIELD_DEF(uint32_t, seq_len);          // S
    TILING_DATA_FIELD_DEF(uint32_t, hidden);           // D
  
    // attrs
    TILING_DATA_FIELD_DEF(uint32_t, num_heads);        // H
    TILING_DATA_FIELD_DEF(uint32_t, num_kv_heads);     // H_kv
  
    // tiling: 每个 block 处理多少个 (b, s) token
    TILING_DATA_FIELD_DEF(uint32_t, tokens_per_block); // tp
  END_TILING_DATA_DEF;
  
  REGISTER_TILING_DATA_CLASS(FusedQKVProjectTranspose, FusedQKVProjectTransposeTilingData)
  
}  // namespace optiling