/*
 * GPU regex matching - Kokkos port
 *
 * Ported from grep-omp. NFA is built on CPU using pointer-based structures,
 * then flattened to integer-indexed arrays and uploaded to device for parallel
 * line matching via Kokkos::parallel_for.
 *
 * Original NFA implementation by Russ Cox (MIT License).
 */

#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/time.h>
#include <chrono>

// ============================================================
// NFA constants (from nfautil.h)
// ============================================================
#define LINE_SIZE 200
#define ANY         0x15
#define CONCATENATE 0x1b
#define ALTERNATE   0x04
#define QUESTION    0x02
#define STAR        0x03
#define PLUS        0x01
#define PAREN_OPEN  0x05
#define PAREN_CLOSE 0x06
#define BUFFER_SIZE 8000

typedef unsigned int u32;

typedef struct State State;
struct State {
    int c, id;
    State *out, *out1, *dev;
    int lastlist;
    unsigned char free;
};

typedef union Ptrlist Ptrlist;
union Ptrlist { Ptrlist *next; State *s; };

typedef struct Frag Frag;
struct Frag { State *start; Ptrlist *out; };

enum { Match = 256, Split = 257, Any = 258 };

// ============================================================
// Regex simplifier (from regex.h/regex.cpp)
// ============================================================
typedef struct {
    char *re;
    int size;
} SimpleReBuilder;

#define DEREF(arr,i) ((*(arr))[(i)])

static void simpleReBuilder(SimpleReBuilder **builder, int len) {
    (*builder)->re = (char *)malloc(len + 3);
    (*builder)->size = len + 3;
}
static void _simpleReBuilder(SimpleReBuilder *builder) { free(builder->re); }

static void insertIntoComplexRe(char **complexRe, int where, int *len, const char *toInsert) {
    int insertLen = strlen(toInsert);
    int i = where;
    *len = *len + (insertLen + 1);
    *complexRe = (char *)realloc(*complexRe, *len);
    char *buf = (char *)malloc(*len);
    for (int k = i + 2; k < *len - (insertLen + 1); k++) buf[k-(i+2)] = DEREF(complexRe,k);
    for (int k = 0; k < insertLen; k++) DEREF(complexRe, i++) = toInsert[k];
    for (int k = i; k < *len; k++) DEREF(complexRe,k) = buf[k-i];
    free(buf);
}

static void handle_escape(SimpleReBuilder *builder, char **complexRe, int *len, int *bi, int *ci) {
    int i = *ci, j = *bi;
    if (i+1 > *len) { fprintf(stderr,"bad escape\n"); exit(1); }
    i++;
    switch(DEREF(complexRe,i)) {
        case 't': insertIntoComplexRe(complexRe, --i, len, "\t"); break;
        case 'n': insertIntoComplexRe(complexRe, --i, len, "\n"); break;
        case 'd': insertIntoComplexRe(complexRe, --i, len, "[0-9]"); break;
        case 'w': insertIntoComplexRe(complexRe, --i, len, "([a-z]|[A-Z]|_)"); break;
        case 's': insertIntoComplexRe(complexRe, --i, len, "( |\t|\n)"); break;
        default: builder->re[j++] = DEREF(complexRe, i++); break;
    }
    *ci = i - 1; *bi = j - 1;
}

static void putRange(SimpleReBuilder *builder, char start, char end, int *bi) {
    int i = *bi;
    int amount = ((end - start + 1) * 2) + 1;
    builder->size += amount;
    builder->re = (char *)realloc(builder->re, builder->size);
    builder->re[i++] = PAREN_OPEN;
    builder->re[i++] = start;
    for (char k = start+1; k <= end; k++) { builder->re[i++] = ALTERNATE; builder->re[i++] = k; }
    builder->re[i] = PAREN_CLOSE;
    *bi = i;
}

static void handle_range(SimpleReBuilder *builder, char *complexRe, int len, int *bi, int *ci) {
    int i = *ci;
    if (complexRe[i+4] != ']' || complexRe[i+2] != '-' || complexRe[i+1] > complexRe[i+3])
    { fprintf(stderr,"bad range\n"); exit(1); }
    putRange(builder, complexRe[i+1], complexRe[i+3], bi);
    *ci = i + 4;
}

static SimpleReBuilder *simplifyRe(char **complexRe, SimpleReBuilder *builder) {
    int len = strlen(*complexRe);
    simpleReBuilder(&builder, len);
    for (int i = 0, j = 0; i < len; i++, j++) {
        switch(DEREF(complexRe,i)) {
            case '\\': handle_escape(builder, complexRe, &len, &j, &i); break;
            case '.':  builder->re[j] = ANY;        break;
            case '+':  builder->re[j] = PLUS;       break;
            case '?':  builder->re[j] = QUESTION;   break;
            case '*':  builder->re[j] = STAR;        break;
            case '|':  builder->re[j] = ALTERNATE;  break;
            case '(':  builder->re[j] = PAREN_OPEN;  break;
            case ')':  builder->re[j] = PAREN_CLOSE; break;
            case '[':  handle_range(builder, *complexRe, len, &j, &i); break;
            default:   builder->re[j] = DEREF(complexRe,i); break;
        }
    }
    builder->re[strlen(builder->re)+1] = '\0';
    return builder;
}

static char *stringifyRegex(const char *oldRegex) {
    int len = strlen(oldRegex);
    char *n = (char *)malloc(len+1);
    for (int i = 0; i < len; i++) {
        switch(oldRegex[i]) {
            case ANY:        n[i] = '.'; break;
            case CONCATENATE:n[i] = '`'; break;
            case ALTERNATE:  n[i] = '|'; break;
            case QUESTION:   n[i] = '?'; break;
            case STAR:       n[i] = '*'; break;
            case PLUS:       n[i] = '+'; break;
            case PAREN_OPEN: n[i] = '('; break;
            case PAREN_CLOSE:n[i] = ')'; break;
            default:         n[i] = oldRegex[i]; break;
        }
    }
    n[len] = '\0';
    return n;
}

// ============================================================
// CPU-side NFA construction (re2post, pstate, ppost2nfa, pre2post)
// ============================================================

// Convert infix regex to postfix (CPU version, static buffer)
static char *re2post(char *re) {
    static char buf[8000];
    char *dst = buf;
    int nalt = 0, natom = 0;
    struct { int nalt, natom; } paren[100], *p = paren;
    if (strlen(re) >= sizeof(buf)/2) return NULL;
    for (; *re; re++) {
        switch(*re) {
            case PAREN_OPEN:
                if (natom > 1) { --natom; *dst++ = CONCATENATE; }
                if (p >= paren+100) return NULL;
                p->nalt = nalt; p->natom = natom; p++; nalt = 0; natom = 0; break;
            case ALTERNATE:
                if (natom == 0) return NULL;
                while (--natom > 0) *dst++ = CONCATENATE;
                nalt++; break;
            case PAREN_CLOSE:
                if (p == paren || natom == 0) return NULL;
                while (--natom > 0) *dst++ = CONCATENATE;
                for (; nalt > 0; nalt--) *dst++ = ALTERNATE;
                --p; nalt = p->nalt; natom = p->natom; natom++; break;
            case STAR: case PLUS: case QUESTION:
                if (natom == 0) return NULL;
                *dst++ = *re; break;
            default:
                if (natom > 1) { --natom; *dst++ = CONCATENATE; }
                *dst++ = *re; natom++; break;
        }
    }
    if (p != paren) return NULL;
    while (--natom > 0) *dst++ = CONCATENATE;
    for (; nalt > 0; nalt--) *dst++ = ALTERNATE;
    *dst = 0;
    return buf;
}

// Inline device-compatible version (writes to caller-provided dst)
static inline char *pre2post(char *re, char *dst) {
    int nalt = 0, natom = 0;
    struct { int nalt, natom; } paren[100], *p = paren;
    int len = 0; { char *t = re; while (*t++) len++; }
    if (len >= BUFFER_SIZE/2) return NULL;
    for (; *re; re++) {
        switch(*re) {
            case PAREN_OPEN:
                if (natom > 1) { --natom; *dst++ = CONCATENATE; }
                if (p >= paren+100) return NULL;
                p->nalt = nalt; p->natom = natom; p++; nalt = 0; natom = 0; break;
            case ALTERNATE:
                if (natom == 0) return NULL;
                while (--natom > 0) *dst++ = CONCATENATE;
                nalt++; break;
            case PAREN_CLOSE:
                if (p == paren || natom == 0) return NULL;
                while (--natom > 0) *dst++ = CONCATENATE;
                for (; nalt > 0; nalt--) *dst++ = ALTERNATE;
                --p; nalt = p->nalt; natom = p->natom; natom++; break;
            case STAR: case PLUS: case QUESTION:
                if (natom == 0) return NULL;
                *dst++ = *re; break;
            default:
                if (natom > 1) { --natom; *dst++ = CONCATENATE; }
                *dst++ = *re; natom++; break;
        }
    }
    if (p != paren) return NULL;
    while (--natom > 0) *dst++ = CONCATENATE;
    for (; nalt > 0; nalt--) *dst++ = ALTERNATE;
    *dst = 0;
    return dst;
}

// Ptrlist helpers
static inline Frag pfrag(State *start, Ptrlist *out) { Frag n = { start, out }; return n; }
static inline Ptrlist *plist1(State **outp) { Ptrlist *l = (Ptrlist *)outp; l->next = NULL; return l; }
static inline void ppatch(Ptrlist *l, State *s) { for (Ptrlist *next; l; l = next) { next = l->next; l->s = s; } }
static inline Ptrlist *pappend(Ptrlist *l1, Ptrlist *l2) {
    Ptrlist *old = l1;
    while (l1->next) l1 = l1->next;
    l1->next = l2; return old;
}

static inline State *pstate(int c, State *out, State *out1, State *lstate, int *pnstate) {
    State *s = lstate + *pnstate;
    s->id = *pnstate; (*pnstate)++;
    s->lastlist = 0; s->c = c; s->out = out; s->out1 = out1;
    s->dev = NULL; s->free = 0;
    return s;
}

static inline State *ppost2nfa(char *postfix, State *lstate, int *pnstate, State *pmatchstate) {
#define ppush(s) *stackp++ = s
#define ppop()  *--stackp
    if (!postfix) return NULL;
    Frag stack[1000], *stackp = stack, e1, e2, e;
    State *s;
    for (char *p = postfix; *p; p++) {
        switch(*p) {
            case ANY:
                s = pstate(Any, NULL, NULL, lstate, pnstate);
                ppush(pfrag(s, plist1(&s->out))); break;
            default:
                s = pstate(*p, NULL, NULL, lstate, pnstate);
                ppush(pfrag(s, plist1(&s->out))); break;
            case CONCATENATE:
                e2 = ppop(); e1 = ppop();
                ppatch(e1.out, e2.start);
                ppush(pfrag(e1.start, e2.out)); break;
            case ALTERNATE:
                e2 = ppop(); e1 = ppop();
                s = pstate(Split, e1.start, e2.start, lstate, pnstate);
                ppush(pfrag(s, pappend(e1.out, e2.out))); break;
            case QUESTION:
                e = ppop();
                s = pstate(Split, e.start, NULL, lstate, pnstate);
                ppush(pfrag(s, pappend(e.out, plist1(&s->out1)))); break;
            case STAR:
                e = ppop();
                s = pstate(Split, e.start, NULL, lstate, pnstate);
                ppatch(e.out, s);
                ppush(pfrag(s, plist1(&s->out1))); break;
            case PLUS:
                e = ppop();
                s = pstate(Split, e.start, NULL, lstate, pnstate);
                ppatch(e.out, s);
                ppush(pfrag(e.start, plist1(&s->out1))); break;
        }
    }
    e = ppop();
    if (stackp != stack) return NULL;
    ppatch(e.out, pmatchstate);
    return e.start;
#undef ppush
#undef ppop
}

// ============================================================
// File I/O and command-line parsing (from nfautil.cpp)
// ============================================================

static void readFile(char *fileName, char ***lines, int *lineIndex) {
    FILE *fp = fopen(fileName, "r");
    char *source = NULL;
    if (fp) {
        fseek(fp, 0L, SEEK_END);
        long bufsize = ftell(fp);
        source = (char *)malloc(sizeof(char) * (bufsize + 1));
        fseek(fp, 0L, SEEK_SET);
        size_t newLen = fread(source, sizeof(char), bufsize, fp);
        if (newLen == 0) fputs("Error reading file", stderr);
        else source[newLen] = '\0';
        fclose(fp);
    }
    *lines = (char **)malloc(sizeof(char *));
    **lines = source;
    *lineIndex = 1;
}

static void usage(const char *progname) {
    printf("Usage: %s [options] [pattern]\n", progname);
    printf("  -v  Visualize NFA\n  -p  Show postfix\n  -s  Show simplified\n");
    printf("  -t  Print timing\n  -f <FILE> Input file\n");
}

static void parseCmdLine(int argc, char **argv, int *visualize, int *postfix,
                         int *time, int *simplified, char **fileName, char **regexFile) {
    if (argc < 3) { usage(argv[0]); exit(EXIT_SUCCESS); }
    static struct option long_options[] = {
        {"help",0,0,'?'},{"postfix",0,0,'p'},{"simplified",0,0,'s'},
        {"visualize",0,0,'v'},{"file",1,0,'f'},{"regex",1,0,'r'},{"time",0,0,'t'},{0,0,0,0}
    };
    *visualize = *postfix = *time = *simplified = 0;
    int opt;
    while ((opt = getopt_long_only(argc, argv, "tvpsf:r:?", long_options, NULL)) != EOF) {
        switch(opt) {
            case 'v': *visualize = 1; break;
            case 'p': *postfix = 1;  break;
            case 'f': *fileName = optarg; break;
            case 'r': *regexFile = optarg; break;
            case 't': *time = 1; break;
            case 's': *simplified = 1; break;
            default: usage(argv[0]); exit(EXIT_SUCCESS);
        }
    }
}

// ============================================================
// NFA flattening: pointer-based → integer-indexed
// ============================================================
#define MAX_NFA_STATES 104

struct FlatNFA {
    int c[MAX_NFA_STATES];   // character class: Match/Split/Any/<char>
    int out[MAX_NFA_STATES]; // index of out state, -1=NULL, fn=match
    int out1[MAX_NFA_STATES];
    int start;               // start state index
    int fn;                  // number of real states; match state index = fn
};

static void flatten_nfa(State *lstate, int nstates, State *pmatch, State *start_st,
                         FlatNFA &flat) {
    flat.fn = nstates;
    flat.start = (int)(start_st - lstate);

    auto state_to_idx = [&](State *s) -> int {
        if (!s) return -1;
        if (s == pmatch) return nstates;  // match state
        return (int)(s - lstate);
    };

    // Populate real states
    for (int i = 0; i < nstates; i++) {
        flat.c[i]    = lstate[i].c;
        flat.out[i]  = state_to_idx(lstate[i].out);
        flat.out1[i] = state_to_idx(lstate[i].out1);
    }
    // Match state at index fn
    flat.c[nstates]    = Match;
    flat.out[nstates]  = -1;
    flat.out1[nstates] = -1;
}

// ============================================================
// Device-side NFA matching using flat integer arrays
// ============================================================

KOKKOS_INLINE_FUNCTION
static void flat_epsilon_close(int idx, int *list, int &n,
                                bool *visited,
                                const int *fc, const int *fo, const int *fo1)
{
    // Iterative DFS to follow Split edges
    int stk[MAX_NFA_STATES + 2];
    int stk_n = 0;
    stk[stk_n++] = idx;
    while (stk_n > 0) {
        int s = stk[--stk_n];
        if (s < 0 || visited[s]) continue;
        visited[s] = true;
        if (fc[s] == Split) {
            if (fo[s]  >= 0) stk[stk_n++] = fo[s];
            if (fo1[s] >= 0) stk[stk_n++] = fo1[s];
        } else {
            list[n++] = s;
        }
    }
}

KOKKOS_INLINE_FUNCTION
static bool flat_ismatch(const int *list, int n, const int *fc) {
    for (int i = 0; i < n; i++)
        if (fc[list[i]] == Match) return true;
    return false;
}

KOKKOS_INLINE_FUNCTION
static bool flat_match_line(int fstart, int fn,
                             const int *fc, const int *fo, const int *fo1,
                             const char *line_ptr)
{
    int list0[MAX_NFA_STATES + 2], list1[MAX_NFA_STATES + 2];
    int n0 = 0, n1 = 0;
    bool visited[MAX_NFA_STATES + 2];

    // Epsilon closure of start
    for (int i = 0; i <= fn; i++) visited[i] = false;
    flat_epsilon_close(fstart, list0, n0, visited, fc, fo, fo1);

    for (int ci = 0; line_ptr[ci] != '\0'; ci++) {
        int c = (unsigned char)line_ptr[ci];
        n1 = 0;
        for (int i = 0; i <= fn; i++) visited[i] = false;
        for (int i = 0; i < n0; i++) {
            int s = list0[i];
            if (fc[s] == Match) continue;
            if (fc[s] == c || fc[s] == Any) {
                int next = fo[s];
                if (next >= 0) flat_epsilon_close(next, list1, n1, visited, fc, fo, fo1);
            }
        }
        // swap
        for (int i = 0; i < n1; i++) list0[i] = list1[i];
        n0 = n1;
    }
    return flat_ismatch(list0, n0, fc);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv)
{
    int timerOn;
    char *fileName = NULL, *regexFile = NULL, **lines = NULL;
    int num_lines;

    SimpleReBuilder builder;
    int visualize, simplified, postfix_opt;

    parseCmdLine(argc, argv, &visualize, &postfix_opt, &timerOn, &simplified, &fileName, &regexFile);

    int regexIndex = 1 + visualize + postfix_opt + timerOn + simplified;
    if (fileName != NULL) regexIndex += 2;
    if (argc <= regexIndex) { usage(argv[0]); return EXIT_SUCCESS; }

    // Simplify regex
    char *regexBuffer = (char *)malloc(strlen(argv[regexIndex]) + 1);
    strcpy(regexBuffer, argv[regexIndex]);
    simplifyRe(&regexBuffer, &builder);
    free(regexBuffer);

    char *post = re2post(builder.re);
    if (!post) { fprintf(stderr, "bad regexp: %s\n", argv[regexIndex]); return 1; }

    if (simplified) {
        char *s = stringifyRegex(builder.re); printf("\nSimplified Regex: %s\n", s); free(s);
        _simpleReBuilder(&builder); return 0;
    }
    if (postfix_opt) {
        char *s = stringifyRegex(post); printf("\nPostfix buffer: %s\n", s); free(s);
        _simpleReBuilder(&builder); return 0;
    }

    if (!fileName) { printf("Enter a file\n"); return EXIT_SUCCESS; }

    auto t0 = std::chrono::steady_clock::now();
    readFile(fileName, &lines, &num_lines);
    auto t1 = std::chrono::steady_clock::now();

    // Build flat NFA on CPU
    char postbuf[BUFFER_SIZE];
    pre2post(builder.re, postbuf);

    State s[100];
    int pnstate = 0;
    State pmatchstate = { Match };
    State *st = ppost2nfa(postbuf, s, &pnstate, &pmatchstate);
    if (!st) { fprintf(stderr, "NFA construction failed\n"); return 1; }

    FlatNFA flat;
    flatten_nfa(s, pnstate, &pmatchstate, st, flat);

    if (visualize) { printf("NFA has %d states, start=%d\n", flat.fn, flat.start); return 0; }

    // Build line table (null-terminate lines)
    u32 *table = (u32 *)malloc(sizeof(u32) * strlen(*lines) + sizeof(u32));
    table[0] = 0;
    num_lines = 0;
    int len = strlen(lines[0]);
    for (int i = 0; i < len; i++) {
        if ((lines[0])[i] == '\n') {
            table[++num_lines] = i + 1;
            lines[0][i] = '\0';
        }
    }
    if (len > 0 && (lines[0])[len-1] == '\n') --num_lines;

    auto t2 = std::chrono::steady_clock::now();

    Kokkos::initialize(argc, argv);
    {
        int flat_nstates = flat.fn;
        int flat_start   = flat.start;

        // Upload flat NFA to device
        Kokkos::View<int*> d_fc  ("fc",  flat_nstates + 1);
        Kokkos::View<int*> d_fo  ("fo",  flat_nstates + 1);
        Kokkos::View<int*> d_fo1 ("fo1", flat_nstates + 1);
        {
            auto hfc  = Kokkos::create_mirror_view(d_fc);
            auto hfo  = Kokkos::create_mirror_view(d_fo);
            auto hfo1 = Kokkos::create_mirror_view(d_fo1);
            for (int i = 0; i <= flat_nstates; i++) {
                hfc(i)  = flat.c[i];
                hfo(i)  = flat.out[i];
                hfo1(i) = flat.out1[i];
            }
            Kokkos::deep_copy(d_fc,  hfc);
            Kokkos::deep_copy(d_fo,  hfo);
            Kokkos::deep_copy(d_fo1, hfo1);
        }

        // Upload line buffer and table
        Kokkos::View<char*>  d_line ("line",  len + 1);
        Kokkos::View<int*>   d_table("table", num_lines + 1);
        {
            auto hl = Kokkos::create_mirror_view(d_line);
            auto ht = Kokkos::create_mirror_view(d_table);
            for (int i = 0; i <= len; i++) hl(i) = lines[0][i];
            for (int i = 0; i <= num_lines; i++) ht(i) = (int)table[i];
            Kokkos::deep_copy(d_line,  hl);
            Kokkos::deep_copy(d_table, ht);
        }

        Kokkos::View<unsigned char*> d_result("result", num_lines);

        auto t3 = std::chrono::steady_clock::now();

        Kokkos::parallel_for("match_lines",
            Kokkos::RangePolicy<>(0, num_lines),
            KOKKOS_LAMBDA(int i) {
                const int *fc  = d_fc.data();
                const int *fo  = d_fo.data();
                const int *fo1 = d_fo1.data();
                const char *line_ptr = d_line.data() + d_table(i);
                d_result(i) = flat_match_line(flat_start, flat_nstates,
                                              fc, fo, fo1, line_ptr) ? 1 : 0;
            });
        Kokkos::fence();

        auto t4 = std::chrono::steady_clock::now();

        // Copy result back
        auto h_result = Kokkos::create_mirror_view(d_result);
        Kokkos::deep_copy(h_result, d_result);

        if (!timerOn) {
            for (int i = 0; i < num_lines; i++)
                if (h_result(i)) printf("%s\n", lines[0] + table[i]);
        }

        auto t5 = std::chrono::steady_clock::now();

        if (timerOn) {
            auto us = [](auto a, auto b) {
                return std::chrono::duration_cast<std::chrono::microseconds>(b-a).count() / 1e6;
            };
            printf("\nReadFile time %.4f\n",     us(t0, t1));
            printf("Device setup time %.4f\n",   us(t2, t3));
            printf("Kernel execution time %.4f\n",us(t3, t4));
            printf("Total time %.4f\n\n",         us(t0, t5));
        }
    }
    Kokkos::finalize();

    free(table);
    free(*lines);
    free(lines);
    _simpleReBuilder(&builder);
    return EXIT_SUCCESS;
}
