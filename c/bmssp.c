#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define INF DBL_MAX

// -------- Graph --------
typedef struct { int to; double w; } Edge;

typedef struct {
    Edge *edges;
    int count;
    int capacity;
} EdgeList;

typedef struct {
    int n;
    EdgeList *adj;
} Graph;

Graph* new_graph(int n) {
    Graph *g = (Graph*)malloc(sizeof(Graph));
    g->n = n;
    g->adj = (EdgeList*)calloc(n, sizeof(EdgeList));
    return g;
}

void add_edge(Graph *g, int u, int v, double w) {
    if (w < 0) {
        fprintf(stderr, "Only non-negative weights allowed\n");
        exit(1);
    }
    EdgeList *list = &g->adj[u];
    if (list->count == list->capacity) {
        list->capacity = list->capacity ? list->capacity * 2 : 4;
        list->edges = (Edge*)realloc(list->edges, list->capacity * sizeof(Edge));
    }
    list->edges[list->count++] = (Edge){v, w};
    // ensure adjacency for v exists
    if (!g->adj[v].edges && g->adj[v].count == 0) {
        g->adj[v].edges = NULL;
        g->adj[v].capacity = 0;
    }
}

// -------- IntArray helper (used as sets) --------
typedef struct {
    int *data;
    int count;
    int capacity;
} IntArray;

void ia_init(IntArray *a) {
    a->data = NULL;
    a->count = 0;
    a->capacity = 0;
}

void ia_free(IntArray *a) {
    free(a->data);
}

int ia_contains(const IntArray *a, int x) {
    for (int i = 0; i < a->count; i++) if (a->data[i] == x) return 1;
    return 0;
}

void ia_add_unique(IntArray *a, int x) {
    if (ia_contains(a, x)) return;
    if (a->count == a->capacity) {
        a->capacity = a->capacity ? a->capacity * 2 : 4;
        a->data = (int*)realloc(a->data, a->capacity * sizeof(int));
    }
    a->data[a->count++] = x;
}

void ia_copy(IntArray *dest, const IntArray *src) {
    dest->count = 0;
    dest->capacity = src->count;
    dest->data = (int*)realloc(dest->data, dest->capacity * sizeof(int));
    for (int i = 0; i < src->count; i++) dest->data[dest->count++] = src->data[i];
}

// -------- MinHeap --------
typedef struct { double dist; int node; } NodeDist;

typedef struct {
    NodeDist *data;
    int size;
    int capacity;
} MinHeap;

void heap_init(MinHeap *h) {
    h->data = NULL;
    h->size = 0;
    h->capacity = 0;
}

void heap_free(MinHeap *h) {
    free(h->data);
}

void heap_swap(NodeDist *a, NodeDist *b) {
    NodeDist tmp = *a; *a = *b; *b = tmp;
}

void heap_push(MinHeap *h, NodeDist nd) {
    if (h->size == h->capacity) {
        h->capacity = h->capacity ? h->capacity * 2 : 8;
        h->data = (NodeDist*)realloc(h->data, h->capacity * sizeof(NodeDist));
    }
    int i = h->size++;
    h->data[i] = nd;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->data[parent].dist <= h->data[i].dist) break;
        heap_swap(&h->data[parent], &h->data[i]);
        i = parent;
    }
}

NodeDist heap_pop(MinHeap *h) {
    NodeDist res = h->data[0];
    h->data[0] = h->data[--h->size];
    int i = 0;
    while (1) {
        int left = 2*i + 1, right = 2*i + 2, smallest = i;
        if (left < h->size && h->data[left].dist < h->data[smallest].dist) smallest = left;
        if (right < h->size && h->data[right].dist < h->data[smallest].dist) smallest = right;
        if (smallest == i) break;
        heap_swap(&h->data[i], &h->data[smallest]);
        i = smallest;
    }
    return res;
}

int heap_empty(MinHeap *h) { return h->size == 0; }

// -------- DQueue --------
typedef struct {
    int M;
    double B;
    MinHeap data;
    NodeDist *prepend;
    int pre_count;
    int pre_capacity;
} DQueue;

void dq_init(DQueue *dq, int M, double B) {
    dq->M = M; dq->B = B;
    heap_init(&dq->data);
    dq->prepend = NULL; dq->pre_count = 0; dq->pre_capacity = 0;
}

void dq_free(DQueue *dq) {
    heap_free(&dq->data);
    free(dq->prepend);
}

void dq_insert(DQueue *dq, int node, double dist) {
    if (dist >= dq->B) return;
    heap_push(&dq->data, (NodeDist){dist, node});
}

void dq_batch_prepend(DQueue *dq, NodeDist *items, int count) {
    for (int i = 0; i < count; i++) {
        if (items[i].dist < dq->B) {
            if (dq->pre_count == dq->pre_capacity) {
                dq->pre_capacity = dq->pre_capacity ? dq->pre_capacity * 2 : 8;
                dq->prepend = (NodeDist*)realloc(dq->prepend, dq->pre_capacity * sizeof(NodeDist));
            }
            dq->prepend[dq->pre_count++] = items[i];
        }
    }
}

int dq_non_empty(DQueue *dq) {
    return dq->pre_count > 0 || !heap_empty(&dq->data);
}

void dq_pull(DQueue *dq, double *Bi, IntArray *Si) {
    ia_init(Si);
    if (dq->pre_count > 0) {
        int take = dq->pre_count < dq->M ? dq->pre_count : dq->M;
        *Bi = dq->prepend[0].dist;
        for (int i = 0; i < take; i++) ia_add_unique(Si, dq->prepend[i].node);
        dq->pre_count -= take;
        for (int i = 0; i < dq->pre_count; i++) dq->prepend[i] = dq->prepend[i+take];
        return;
    }
    if (heap_empty(&dq->data)) { *Bi = dq->B; return; }
    NodeDist *tmp = (NodeDist*)malloc(dq->M * sizeof(NodeDist));
    int cnt = 0;
    while (!heap_empty(&dq->data) && cnt < dq->M) tmp[cnt++] = heap_pop(&dq->data);
    *Bi = cnt ? tmp[0].dist : dq->B;
    for (int i = 0; i < cnt; i++) ia_add_unique(Si, tmp[i].node);
    free(tmp);
}

// -------- Algorithm 1: FindPivots --------
int subtree_size_rec(int u, IntArray *children, int *seen) {
    if (seen[u]) return 0;
    seen[u] = 1;
    int size = 1;
    for (int i = 0; i < children[u].count; i++) {
        size += subtree_size_rec(children[u].data[i], children, seen);
    }
    seen[u] = 0;
    return size;
}

void find_pivots(Graph *graph, double B, IntArray *S, double *d_hat, IntArray *complete, int k, IntArray *P_out, IntArray *W_out) {
    IntArray W; ia_init(&W); ia_copy(&W, S);
    IntArray W_prev; ia_init(&W_prev); ia_copy(&W_prev, S);
    for (int iter = 1; iter <= k; iter++) {
        IntArray W_i; ia_init(&W_i);
        for (int idx = 0; idx < W_prev.count; idx++) {
            int u = W_prev.data[idx];
            double du = d_hat[u];
            if (du >= B) continue;
            EdgeList *lst = &graph->adj[u];
            for (int e = 0; e < lst->count; e++) {
                int v = lst->edges[e].to; double w = lst->edges[e].w;
                if (du + w <= d_hat[v]) {
                    d_hat[v] = du + w;
                    if (du + w < B) ia_add_unique(&W_i, v);
                }
            }
        }
        for (int i = 0; i < W_i.count; i++) ia_add_unique(&W, W_i.data[i]);
        if (W.count > k * S->count) { ia_copy(P_out, S); ia_copy(W_out, &W); ia_free(&W_i); ia_free(&W_prev); ia_free(&W); return; }
        ia_free(&W_prev);
        ia_init(&W_prev); ia_copy(&W_prev, &W_i);
        ia_free(&W_i);
    }
    // build F_children
    IntArray *children = (IntArray*)malloc(graph->n * sizeof(IntArray));
    int *indeg = (int*)calloc(graph->n, sizeof(int));
    for (int i = 0; i < graph->n; i++) ia_init(&children[i]);
    for (int idx = 0; idx < W.count; idx++) {
        int u = W.data[idx]; double du = d_hat[u];
        EdgeList *lst = &graph->adj[u];
        for (int e = 0; e < lst->count; e++) {
            int v = lst->edges[e].to; double w = lst->edges[e].w;
            if (ia_contains(&W, v) && fabs(d_hat[v] - (du + w)) < 1e-12) {
                ia_add_unique(&children[u], v);
                indeg[v]++;
            }
        }
    }

    ia_init(P_out);
    int *seen = (int*)calloc(graph->n, sizeof(int));
    for (int idx = 0; idx < S->count; idx++) {
        int u = S->data[idx];
        if (indeg[u] == 0 && subtree_size_rec(u, children, seen) >= k) ia_add_unique(P_out, u);
    }
    free(seen);
    ia_copy(W_out, &W);
    for (int i = 0; i < graph->n; i++) ia_free(&children[i]);
    free(children); free(indeg); ia_free(&W_prev); ia_free(&W);
}

// -------- Algorithm 2: BaseCase --------
void base_case(Graph *graph, double B, IntArray *S, double *d_hat, IntArray *complete, int k, double *B_out, IntArray *U_out) {
    int x = S->data[0];
    IntArray U0; ia_init(&U0); ia_add_unique(&U0, x);
    MinHeap H; heap_init(&H); heap_push(&H, (NodeDist){d_hat[x], x});
    int *visited = (int*)calloc(graph->n, sizeof(int));
    while (!heap_empty(&H) && U0.count < k + 1) {
        NodeDist nd = heap_pop(&H); int u = nd.node; double du = nd.dist;
        if (visited[u]) continue; visited[u] = 1; ia_add_unique(&U0, u); ia_add_unique(complete, u);
        EdgeList *lst = &graph->adj[u];
        for (int e = 0; e < lst->count; e++) {
            int v = lst->edges[e].to; double w = lst->edges[e].w;
            if (du + w <= d_hat[v] && du + w < B) {
                d_hat[v] = du + w; heap_push(&H, (NodeDist){d_hat[v], v});
            }
        }
    }
    if (U0.count <= k) { *B_out = B; ia_copy(U_out, &U0); }
    else {
        double Bp = -INFINITY; for (int i = 0; i < U0.count; i++) if (d_hat[U0.data[i]] > Bp) Bp = d_hat[U0.data[i]];
        *B_out = Bp; ia_init(U_out); for (int i = 0; i < U0.count; i++) if (d_hat[U0.data[i]] < Bp) ia_add_unique(U_out, U0.data[i]);
    }
    free(visited); ia_free(&U0); heap_free(&H);
}

// -------- Algorithm 3: BMSSP --------
void bmssp(Graph *graph, int l, double B, IntArray *S, double *d_hat, IntArray *complete, int k, int t, double *B_out, IntArray *U_out) {
    ia_init(U_out);
    if (l == 0) { base_case(graph, B, S, d_hat, complete, k, B_out, U_out); return; }
    IntArray P, W; ia_init(&P); ia_init(&W);
    find_pivots(graph, B, S, d_hat, complete, k, &P, &W);
    int M = 1 << ((l - 1) * t);
    DQueue D; dq_init(&D, M, B);
    for (int i = 0; i < P.count; i++) dq_insert(&D, P.data[i], d_hat[P.data[i]]);
    IntArray U; ia_init(&U);
    double B0_prime = B; for (int i = 0; i < P.count; i++) if (d_hat[P.data[i]] < B0_prime) B0_prime = d_hat[P.data[i]];
    while (U.count < k * (1 << (l * t)) && dq_non_empty(&D)) {
        double Bi; IntArray Si; dq_pull(&D, &Bi, &Si);
        double B_prime_i; IntArray U_i; ia_init(&U_i); bmssp(graph, l - 1, Bi, &Si, d_hat, complete, k, t, &B_prime_i, &U_i);
        for (int j = 0; j < U_i.count; j++) ia_add_unique(&U, U_i.data[j]);
        IntArray K; ia_init(&K);
        for (int j = 0; j < U_i.count; j++) {
            int u = U_i.data[j]; EdgeList *lst = &graph->adj[u];
            for (int e = 0; e < lst->count; e++) {
                int v = lst->edges[e].to; double w = lst->edges[e].w;
                if (d_hat[u] + w <= d_hat[v]) {
                    d_hat[v] = d_hat[u] + w;
                    if (Bi <= d_hat[v] && d_hat[v] < B) dq_insert(&D, v, d_hat[v]);
                    else if (B_prime_i <= d_hat[v] && d_hat[v] < Bi) ia_add_unique(&K, v);
                }
            }
        }
        NodeDist *prepend = NULL; int pc = 0;
        if (K.count + Si.count) {
            prepend = (NodeDist*)malloc((K.count + Si.count) * sizeof(NodeDist));
            for (int j = 0; j < K.count; j++) prepend[pc++] = (NodeDist){d_hat[K.data[j]], K.data[j]};
            for (int j = 0; j < Si.count; j++) {
                int x = Si.data[j]; double dist = d_hat[x];
                if (B_prime_i <= dist && dist < Bi) prepend[pc++] = (NodeDist){dist, x};
            }
            dq_batch_prepend(&D, prepend, pc);
            free(prepend);
        }
        ia_free(&K); ia_free(&Si); ia_free(&U_i);
    }
    double B_prime = B0_prime < B ? B0_prime : B;
    for (int i = 0; i < W.count; i++) {
        int x = W.data[i]; if (d_hat[x] < B_prime) ia_add_unique(&U, x);
    }
    for (int i = 0; i < U.count; i++) ia_add_unique(complete, U.data[i]);
    *B_out = B_prime; ia_copy(U_out, &U);
    ia_free(&U); dq_free(&D); ia_free(&P); ia_free(&W);
}

// -------- Main Driver --------
void run_sssp(Graph *graph, int source, double *d_hat) {
    int n = graph->n;
    double log2n = log2((double)n);
    int k = (int)fmax(1.0, pow(log2n, 1.0/3.0));
    int t = (int)fmax(1.0, pow(log2n, 2.0/3.0));
    int l = (int)ceil(log2n / t);
    for (int i = 0; i < n; i++) d_hat[i] = INF;
    d_hat[source] = 0.0;
    IntArray complete; ia_init(&complete); ia_add_unique(&complete, source);
    IntArray S; ia_init(&S); ia_add_unique(&S, source);
    double B_out; IntArray U_out; ia_init(&U_out); bmssp(graph, l, INF, &S, d_hat, &complete, k, t, &B_out, &U_out);
    ia_free(&complete); ia_free(&S); ia_free(&U_out);
}

int main() {
    // Example graph with nodes: 0=s,1=a,2=b,3=c
    Graph *g = new_graph(4);
    add_edge(g, 0, 1, 1);
    add_edge(g, 0, 2, 4);
    add_edge(g, 1, 2, 2);
    add_edge(g, 1, 3, 5);
    add_edge(g, 2, 3, 1);
    double dist[4];
    run_sssp(g, 0, dist);
    for (int i = 0; i < 4; i++) {
        if (dist[i] == INF) printf("inf\n");
        else printf("%g\n", dist[i]);
    }
    // free graph
    for (int i = 0; i < g->n; i++) free(g->adj[i].edges);
    free(g->adj); free(g);
    return 0;
}

