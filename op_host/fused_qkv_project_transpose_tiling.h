#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
  BEGIN_TILING_DATA_DEF(TilingData)
    // hidden_states: [B, S, D]
    TILING_DATA_FIELD_DEF(uint32_t, batch);            // B
    TILING_DATA_FIELD_DEF(uint32_t, seq_len);          // S
    TILING_DATA_FIELD_DEF(uint32_t, hidden);           // D
  
    // attrs
    TILING_DATA_FIELD_DEF(uint32_t, num_heads);        // H
    TILING_DATA_FIELD_DEF(uint32_t, num_kv_heads);     // H_kv
    TILING_DATA_FIELD_DEF(uint32_t, head_dim); // Dh
    TILING_DATA_FIELD_DEF(uint32_t, head_kv_dim); // kvDh
    
    TILING_DATA_FIELD_DEF(uint32_t, tokens_per_block); // tp
    TILING_DATA_FIELD_DEF(uint64_t, sys_workspace_size); // hp

    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cube_tiling);

  END_TILING_DATA_DEF;
  
  REGISTER_TILING_DATA_CLASS(FusedQKVProjectTranspose, TilingData)
  
}  // namespace optiling