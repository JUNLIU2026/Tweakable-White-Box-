#include <stdio.h>
#include <stdint.h>
typedef uint16_t u16;

static u16 gf_mul(u16 a, u16 b) {
    uint32_t p = 0, aa = a, bb = b;
    for (int i = 0; i < 16; i++) {
        if (bb & 1) p ^= aa;
        bb >>= 1;
        aa <<= 1;
        if (aa & 0x10000) aa ^= 0x1002B;
    }
    return (u16)(p & 0xFFFF);
}

int main() {
    // 自动找本原元
    u16 g;
    for (g = 2; g < 1000; g++) {
        u16 xx = 1;
        int order = 0;
        for (int i = 0; i < 65535; i++) {
            xx = gf_mul(xx, g);
            order++;
            if (xx == 1) break;
        }
        if (order == 65535) break;
    }
    printf("primitive element = %d\n", g);

    static u16 exp_tab[131072];
    static u16 log_tab[65536];
    u16 x = 1;
    for (int i = 0; i < 65535; i++) {
        exp_tab[i] = x;
        log_tab[x] = i;
        x = gf_mul(x, g);
    }
    for (int i = 65535; i < 131072; i++) exp_tab[i] = exp_tab[i - 65535];

    // 验证
    u16 tests[][2] = {{0x1967,0x1234},{0xFFFF,0x0001},{0x0002,0x8000},{0xABCD,0x1234},{0x0001,0x0001}};
    int ok = 1;
    for (int i = 0; i < 5; i++) {
        u16 a = tests[i][0], b = tests[i][1];
        u16 direct = gf_mul(a, b);
        u16 table = (a==0||b==0) ? 0 : exp_tab[log_tab[a] + log_tab[b]];
        printf("%04X * %04X: direct=%04X table=%04X %s\n",
               a, b, direct, table, direct==table?"OK":"FAIL");
        if (direct != table) ok = 0;
    }
    if (!ok) { printf("FAILED\n"); return 1; }

    FILE *f = fopen("gf_tables.h", "w");
    fprintf(f, "#include <stdint.h>\n\n");
    fprintf(f, "static const uint16_t gf_exp[131072] = {\n");
    for (int i = 0; i < 131072; i++) {
        fprintf(f, "0x%04X,", exp_tab[i]);
        if (i % 16 == 15) fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
    fprintf(f, "static const uint16_t gf_log[65536] = {\n");
    for (int i = 0; i < 65536; i++) {
        fprintf(f, "0x%04X,", log_tab[i]);
        if (i % 16 == 15) fprintf(f, "\n");
    }
    fprintf(f, "};\n");
    fclose(f);
    printf("done, generated gf_tables.h\n");
    return 0;
}
