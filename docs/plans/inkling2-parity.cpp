// Feed llama.cpp explicit token ids; dump last-position logits and named intermediate tensors.
// usage: parity <model.gguf> <out.bin> <id> [id...]
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string g_dump_dir;

// One file per tensor: the whole thing as f32, dims[0] fastest, plus a .dims sidecar.
static bool cb(struct ggml_tensor * t, bool ask, void * /*user*/) {
    if (ask) {
        return true;
    }
    if (g_dump_dir.empty() || t->type != GGML_TYPE_F32) {
        return true;
    }
    const int64_t n = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
    if (n <= 0) {
        return true;
    }
    std::vector<float> v((size_t) n);
    const size_t rs = ggml_row_size(t->type, t->ne[0]);
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                const char * src = (const char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                memcpy(&v[(size_t)((i3 * t->ne[2] + i2) * t->ne[1] + i1) * t->ne[0]], src, (size_t) rs);
            }
        }
    }
    std::string base = g_dump_dir + "/" + t->name;
    FILE * f = fopen((base + ".f32").c_str(), "wb");
    if (f) {
        fwrite(v.data(), sizeof(float), v.size(), f);
        fclose(f);
    }
    FILE * d = fopen((base + ".dims").c_str(), "w");
    if (d) {
        fprintf(d, "%lld %lld %lld %lld\n", (long long) t->ne[0], (long long) t->ne[1],
                (long long) t->ne[2], (long long) t->ne[3]);
        fclose(d);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> <id> [id...]\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    const char * out_path = argv[2];
    std::vector<llama_token> ids;
    for (int i = 3; i < argc; ++i) {
        ids.push_back(atoi(argv[i]));
    }
    if (const char * d = getenv("PARITY_DUMP_DIR")) {
        g_dump_dir = d;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) ids.size() + 8;
    cparams.n_batch = (uint32_t) ids.size();
    cparams.n_ubatch = (uint32_t) ids.size();
    cparams.n_threads = 12;
    cparams.n_threads_batch = 12;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval = cb;
    cparams.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_init((int32_t) ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = (int32_t) ids.size();
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    FILE * f = fopen(out_path, "wb");
    int32_t header[2] = { n_vocab, (int32_t) ids.size() };
    fwrite(header, sizeof(header), 1, f);
    for (size_t i = 0; i < ids.size(); ++i) {
        const float * lg = llama_get_logits_ith(ctx, (int32_t) i);
        fwrite(lg, sizeof(float), (size_t) n_vocab, f);
    }
    fclose(f);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "%zu tokens, %d logits each -> %s\n", ids.size(), n_vocab, out_path);
    return 0;
}
