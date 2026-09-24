/*
 * citysim device code.
 *
 * Each device function has the name and the integer arithmetic of the Rust
 * function it mirrors. gpu::prelude generates every constant and the three
 * struct sizes from the Rust sources and puts them in front of this file.
 */

/** Mirror of `car::Car`. */
struct Car {
    unsigned long long id;
    unsigned int hop;
    unsigned char pos;
    unsigned char vel;
    unsigned char turn;
    unsigned char flags;
};

/** Mirror of `city::NodeMeta`. */
struct NodeMeta {
    unsigned char start[4];
    unsigned char len[4];
    unsigned char phase;
    unsigned char reserved;
    unsigned short elapsed;
    unsigned int sensor;
    unsigned int spawned;
    unsigned int exited;
    unsigned int parked;
    unsigned int crossed;
};

/** Mirror of `gpu::DeviceParams`. */
struct DeviceParams {
    unsigned int rows;
    unsigned int cols;
    unsigned int link_cells;
    unsigned int brake;
    unsigned int turn_straight;
    unsigned int turn_straight_left;
    unsigned int trip_end;
    unsigned int min_green;
    unsigned int max_green;
    unsigned int detector;
    unsigned int key0;
    unsigned int key1;
    unsigned int nodes;
};

static_assert(sizeof(Car) == CAR_BYTES, "Car layout differs from Rust");
static_assert(sizeof(NodeMeta) == NODE_META_BYTES, "NodeMeta layout differs from Rust");
static_assert(sizeof(DeviceParams) == DEVICE_PARAMS_BYTES, "DeviceParams layout differs from Rust");

/** A head car that passes its stop line this tick: `advance_link`'s `HeadRequest`. */
struct HeadRequest {
    bool valid;
    unsigned int from;
    unsigned int vel;
};

/** A head car's wish to leave through `port`, or to park: `junction::Want`. */
struct Want {
    bool valid;
    unsigned char turn;
    unsigned int port;
};

/** Mirror of `philox::round`. */
__device__ __forceinline__ uint4 philox_round(uint4 ctr, uint2 key)
{
    return make_uint4(__umulhi(PHILOX_M1, ctr.z) ^ ctr.y ^ key.x, PHILOX_M1 * ctr.z,
                      __umulhi(PHILOX_M0, ctr.x) ^ ctr.w ^ key.y, PHILOX_M0 * ctr.x);
}

/** Mirror of `philox::philox4x32_10`. */
__device__ uint4 philox4x32_10(uint4 ctr, uint2 key)
{
#pragma unroll
    for (unsigned int r = 0; r < PHILOX_ROUNDS; ++r) {
        if (r > 0) {
            key.x += PHILOX_W0;
            key.y += PHILOX_W1;
        }
        ctr = philox_round(ctr, key);
    }
    return ctr;
}

/** Mirror of `philox::draw_brake`. */
__device__ unsigned int draw_brake(uint2 key, unsigned int spawn_link, unsigned int spawn_tick, unsigned int tick)
{
    return philox4x32_10(make_uint4(spawn_link, spawn_tick, tick, STREAM_BRAKE), key).x;
}

/** Mirror of `philox::draw_route`. */
__device__ uint4 draw_route(uint2 key, unsigned int spawn_link, unsigned int spawn_tick, unsigned int hop)
{
    return philox4x32_10(make_uint4(spawn_link, spawn_tick, hop, STREAM_ROUTE), key);
}

/** Mirror of `philox::draw_spawn`. */
__device__ unsigned int draw_spawn(uint2 key, unsigned int link, unsigned int tick)
{
    return philox4x32_10(make_uint4(link, tick, 0u, STREAM_SPAWN), key).x;
}

/** Mirror of `Car::spawn_link`. */
__device__ unsigned int spawn_link(const Car &car)
{
    return (unsigned int)car.id;
}

/** Mirror of `Car::spawn_tick`. */
__device__ unsigned int spawn_tick(const Car &car)
{
    return (unsigned int)(car.id >> 32);
}

/** Mirror of `Car::spawned`. */
__device__ Car spawned(unsigned int link, unsigned int tick, unsigned char turn)
{
    Car car;
    car.id = ((unsigned long long)tick << 32) | link;
    car.hop = 0;
    car.pos = 0;
    car.vel = 0;
    car.turn = turn;
    car.flags = 0;
    return car;
}

/** Mirror of `car::slot`. */
__device__ unsigned int slot(unsigned int start, unsigned int i, unsigned int capacity)
{
    return (start + i) % capacity;
}

/** Mirror of `grid::opposite`. */
__device__ unsigned int opposite(unsigned int d)
{
    return (d + 2) % 4;
}

/** Mirror of `grid::link_id`. */
__device__ unsigned int link_id(unsigned int node, unsigned int d)
{
    return 4 * node + d;
}

/** Mirror of `grid::neighbour_node`, with -1 for no neighbour. */
__device__ int neighbour_node(unsigned int rows, unsigned int cols, unsigned int node, unsigned int d)
{
    const unsigned int r = node / cols;
    const unsigned int c = node % cols;
    switch (d) {
    case NORTH:
        return r > 0 ? (int)(node - cols) : -1;
    case EAST:
        return c + 1 < cols ? (int)(node + 1) : -1;
    case SOUTH:
        return r + 1 < rows ? (int)(node + cols) : -1;
    default:
        return c > 0 ? (int)(node - 1) : -1;
    }
}

/** Mirror of `grid::exit_port`. */
__device__ unsigned int exit_port(unsigned int approach, unsigned char turn)
{
    switch (turn) {
    case 0:
        return (approach + 2) % 4;
    case 1:
        return (approach + 1) % 4;
    default:
        return (approach + 3) % 4;
    }
}

/** Mirror of `signal::is_green`. */
__device__ bool is_green(unsigned char phase, unsigned int approach)
{
    return approach % 2 == phase;
}

/** Mirror of `signal::next_signal`: writes the phase and green time for the next tick. */
__device__ void next_signal(unsigned char *phase, unsigned short *elapsed, bool green_waiting, bool red_waiting,
                            unsigned int min_green, unsigned int max_green)
{
    const unsigned short held = *elapsed == 0xFFFFu ? *elapsed : (unsigned short)(*elapsed + 1);
    const bool change = held >= min_green && red_waiting && (held >= max_green || !green_waiting);
    if (change) {
        *phase ^= 1;
        *elapsed = 0;
    } else {
        *elapsed = held;
    }
}

/** Mirror of `nasch::nasch_speed`, drawing the braking number only for a car that still moves. */
__device__ unsigned int nasch_speed(unsigned int vel, unsigned int gap, unsigned int brake, uint2 key, const Car &car,
                                    unsigned int tick)
{
    const unsigned int accelerated = min(vel + 1, VMAX);
    const unsigned int safe = min(accelerated, gap);
    if (safe > 0 && draw_brake(key, spawn_link(car), spawn_tick(car), tick) < brake) {
        return safe - 1;
    }
    return safe;
}

/** Mirror of `nasch::advance_link` over the ring `cars` of `capacity` slots. */
__device__ HeadRequest advance_link(Car *cars, unsigned int capacity, unsigned int start, unsigned int len,
                                    unsigned int head_extra, unsigned int brake, uint2 key, unsigned int tick)
{
    const unsigned int stop_line = capacity - 1;
    HeadRequest request = {false, 0, 0};
    unsigned int ahead_old = 0;
    for (unsigned int i = 0; i < len; ++i) {
        Car &car = cars[slot(start, i, capacity)];
        const unsigned int x = car.pos;
        const unsigned int gap = i == 0 ? stop_line - x + head_extra : ahead_old - x - 1;
        const unsigned int v = nasch_speed(car.vel, gap, brake, key, car, tick);
        ahead_old = x;
        if (i == 0 && x + v > stop_line) {
            request.valid = true;
            request.from = x;
            request.vel = v;
        } else {
            car.pos = (unsigned char)(x + v);
            car.vel = (unsigned char)v;
        }
    }
    return request;
}

/** Mirror of `junction::want_of`. */
__device__ Want want_of(unsigned int approach, const Car &car)
{
    Want want;
    want.valid = true;
    want.turn = car.turn;
    want.port = car.turn == PARK ? 0 : exit_port(approach, car.turn);
    return want;
}

/** Mirror of `junction::resolve`. */
__device__ void resolve(const Want wants[4], bool grants[4])
{
    int best[4] = {-1, -1, -1, -1};
    for (unsigned int a = 0; a < 4; ++a) {
        grants[a] = false;
        if (!wants[a].valid) {
            continue;
        }
        if (wants[a].turn == PARK) {
            grants[a] = true;
            continue;
        }
        const int holder = best[wants[a].port];
        if (holder < 0 || wants[a].turn < wants[holder].turn) {
            best[wants[a].port] = (int)a;
        }
    }
    for (unsigned int port = 0; port < 4; ++port) {
        if (best[port] >= 0) {
            grants[best[port]] = true;
        }
    }
}

/** Mirror of `junction::pop_head`. */
__device__ Car pop_head(Car *cars, unsigned int capacity, unsigned char *start, unsigned char *len)
{
    const Car car = cars[*start];
    *start = (unsigned char)((*start + 1) % capacity);
    *len -= 1;
    return car;
}

/** Mirror of `junction::push_tail`. */
__device__ void push_tail(Car *cars, unsigned int capacity, unsigned char start, unsigned char *len, Car car)
{
    cars[slot(start, *len, capacity)] = car;
    *len += 1;
}

/** Mirror of `junction::entry_gap`. */
__device__ unsigned char entry_gap(const Car *cars, unsigned int capacity, unsigned char start, unsigned char len)
{
    return len == 0 ? (unsigned char)capacity : cars[slot(start, len - 1, capacity)].pos;
}

/** Mirror of `city::port_room`. */
__device__ unsigned int port_room(const DeviceParams &p, const unsigned char *gaps, unsigned int node, unsigned int port)
{
    const int next = neighbour_node(p.rows, p.cols, node, port);
    return next < 0 ? VMAX : gaps[link_id((unsigned int)next, opposite(port))];
}

/** Mirror of `city::head_extra`. */
__device__ unsigned int head_extra(const DeviceParams &p, const unsigned char *gaps, unsigned int node, unsigned int d,
                                   const Car &head, bool green)
{
    if (head.turn == PARK) {
        return VMAX;
    }
    if (!green) {
        return 0;
    }
    return port_room(p, gaps, node, want_of(d, head).port);
}

/** Mirror of `city::divert`. */
__device__ unsigned char divert(const DeviceParams &p, const unsigned char *gaps, unsigned int node, unsigned int d,
                                const Car &head, bool green)
{
    const bool at_line = head.pos + 1u == p.link_cells;
    if (!green || head.turn == PARK || head.vel != 0 || !at_line) {
        return head.turn;
    }
    if (port_room(p, gaps, node, want_of(d, head).port) > 0) {
        return head.turn;
    }
    const unsigned char order[3] = {STRAIGHT, LEFT, RIGHT};
    for (unsigned int k = 0; k < 3; ++k) {
        if (order[k] != head.turn && port_room(p, gaps, node, exit_port(d, order[k])) > 0) {
            return order[k];
        }
    }
    return head.turn;
}

/** Mirror of `city::route_turn`. */
__device__ unsigned char route_turn(const DeviceParams &p, uint4 draws)
{
    if (draws.y < p.trip_end) {
        return PARK;
    }
    if (draws.x < p.turn_straight) {
        return STRAIGHT;
    }
    if (draws.x < p.turn_straight_left) {
        return LEFT;
    }
    return RIGHT;
}

/** Mirror of `city::detector_occupied`. */
__device__ bool detector_occupied(const Car *cars, unsigned int capacity, unsigned char start, unsigned char len,
                                  unsigned int detector)
{
    if (len == 0) {
        return false;
    }
    return cars[slot(start, 0, capacity)].pos >= capacity - min(detector, capacity);
}

/** Mirror of `params::spawn_threshold`. */
__device__ unsigned int spawn_threshold(unsigned int base, unsigned int rush, unsigned int boost)
{
    const unsigned long long total = (unsigned long long)base * rush / 100 + boost;
    return total > 0xFFFFFFFFull ? 0xFFFFFFFFu : (unsigned int)total;
}

/** Mirror of `checksum::fnv_mix`. */
__device__ unsigned long long fnv_mix(unsigned long long h, unsigned long long value)
{
    for (unsigned int b = 0; b < 8; ++b) {
        h = (h ^ ((value >> (b * 8)) & 0xFFull)) * FNV_PRIME;
    }
    return h;
}

/** Mirror of `checksum::splitmix64`. */
__device__ unsigned long long splitmix64(unsigned long long x)
{
    unsigned long long z = x + GOLDEN_GAMMA;
    z = (z ^ (z >> 30)) * SPLITMIX_M1;
    z = (z ^ (z >> 27)) * SPLITMIX_M2;
    return z ^ (z >> 31);
}

/** Mirror of `checksum::node_hash`. */
__device__ unsigned long long node_hash(const NodeMeta &m, const Car *cars, unsigned int cells)
{
    unsigned long long h = FNV_OFFSET;
    h = fnv_mix(h, m.phase);
    h = fnv_mix(h, m.elapsed);
    h = fnv_mix(h, m.sensor);
    h = fnv_mix(h, m.spawned);
    h = fnv_mix(h, m.exited);
    h = fnv_mix(h, m.parked);
    h = fnv_mix(h, m.crossed);
    for (unsigned int d = 0; d < 4; ++d) {
        const Car *link = cars + d * cells;
        h = fnv_mix(h, m.len[d]);
        for (unsigned int i = 0; i < m.len[d]; ++i) {
            const Car &car = link[slot(m.start[d], i, cells)];
            h = fnv_mix(h, car.id);
            h = fnv_mix(h, car.hop);
            h = fnv_mix(h, (unsigned long long)car.pos | ((unsigned long long)car.vel << 8) |
                               ((unsigned long long)car.turn << 16));
        }
    }
    return h;
}

/** The node this thread steps, or `p.nodes` past the end. */
__device__ unsigned int this_node(const DeviceParams &p)
{
    const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
    return node < p.nodes ? node : p.nodes;
}

/** Mirror of `city::step_a`, one thread per node. */
extern "C" __global__ void step_a(const DeviceParams p, unsigned int tick, Car *cars, NodeMeta *meta, Car *outbox,
                                  const unsigned char *gaps)
{
    const unsigned int node = this_node(p);
    if (node == p.nodes) {
        return;
    }
    const unsigned int cells = p.link_cells;
    const uint2 key = make_uint2(p.key0, p.key1);
    Car *node_cars = cars + (unsigned long long)node * 4 * cells;
    NodeMeta m = meta[node];
    HeadRequest requests[4];
    Want wants[4];
    for (unsigned int d = 0; d < 4; ++d) {
        requests[d].valid = false;
        wants[d].valid = false;
        if (m.len[d] == 0) {
            continue;
        }
        Car *link = node_cars + d * cells;
        const bool green = is_green(m.phase, d);
        Car &head_slot = link[m.start[d]];
        head_slot.turn = divert(p, gaps, node, d, head_slot, green);
        const Car head = head_slot;
        const unsigned int extra = head_extra(p, gaps, node, d, head, green);
        requests[d] = advance_link(link, cells, m.start[d], m.len[d], extra, p.brake, key, tick);
        if (requests[d].valid) {
            wants[d] = want_of(d, head);
        }
    }
    bool grants[4];
    resolve(wants, grants);
    Car out[4];
    for (unsigned int port = 0; port < 4; ++port) {
        out[port] = spawned(0, 0, 0);
    }
    for (unsigned int d = 0; d < 4; ++d) {
        if (!requests[d].valid || !wants[d].valid) {
            continue;
        }
        Car *link = node_cars + d * cells;
        if (!grants[d]) {
            Car &head = link[m.start[d]];
            head.pos = (unsigned char)(cells - 1);
            head.vel = (unsigned char)(cells - 1 - requests[d].from);
            continue;
        }
        Car car = pop_head(link, cells, &m.start[d], &m.len[d]);
        if (wants[d].turn == PARK) {
            m.parked += 1;
            continue;
        }
        m.sensor += 1;
        m.crossed += 1;
        if (neighbour_node(p.rows, p.cols, node, wants[d].port) < 0) {
            m.exited += 1;
        } else {
            car.pos = (unsigned char)(requests[d].from + requests[d].vel - cells);
            car.vel = (unsigned char)requests[d].vel;
            car.flags = FLAG_VALID;
            out[wants[d].port] = car;
        }
    }
    for (unsigned int port = 0; port < 4; ++port) {
        outbox[4 * node + port] = out[port];
    }
    meta[node] = m;
}

/** Mirror of `city::step_b`, one thread per node. `boost` is the node's event boost, applied when `event_on`. */
extern "C" __global__ void step_b(const DeviceParams p, unsigned int tick, unsigned int rush, unsigned int event_on,
                                  const unsigned int *spawn_base, const unsigned int *event_boost, Car *cars,
                                  NodeMeta *meta, const Car *outbox, unsigned char *gaps)
{
    const unsigned int node = this_node(p);
    if (node == p.nodes) {
        return;
    }
    const unsigned int cells = p.link_cells;
    const uint2 key = make_uint2(p.key0, p.key1);
    Car *node_cars = cars + (unsigned long long)node * 4 * cells;
    NodeMeta m = meta[node];
    const unsigned int boost = event_on ? event_boost[node] : 0;
    for (unsigned int d = 0; d < 4; ++d) {
        Car *link = node_cars + d * cells;
        const int upstream = neighbour_node(p.rows, p.cols, node, d);
        const unsigned int id = link_id(node, d);
        Car incoming = spawned(0, 0, 0);
        if (upstream >= 0) {
            incoming = outbox[4 * upstream + opposite(d)];
        }
        if (incoming.flags & FLAG_VALID) {
            incoming.hop += 1;
            incoming.turn = route_turn(p, draw_route(key, spawn_link(incoming), spawn_tick(incoming), incoming.hop));
            incoming.flags = 0;
            push_tail(link, cells, m.start[d], &m.len[d], incoming);
        } else if (entry_gap(link, cells, m.start[d], m.len[d]) > 0 &&
                   draw_spawn(key, id, tick) < spawn_threshold(spawn_base[id], rush, boost)) {
            const unsigned char turn = route_turn(p, draw_route(key, id, tick, 0));
            push_tail(link, cells, m.start[d], &m.len[d], spawned(id, tick, turn));
            m.spawned += 1;
        }
        gaps[link_id(node, d)] = entry_gap(link, cells, m.start[d], m.len[d]);
    }
    bool green_waiting = false;
    bool red_waiting = false;
    for (unsigned int d = 0; d < 4; ++d) {
        const bool occupied = detector_occupied(node_cars + d * cells, cells, m.start[d], m.len[d], p.detector);
        if (is_green(m.phase, d)) {
            green_waiting = green_waiting || occupied;
        } else {
            red_waiting = red_waiting || occupied;
        }
    }
    next_signal(&m.phase, &m.elapsed, green_waiting, red_waiting, p.min_green, p.max_green);
    meta[node] = m;
}

/** The state of `City::new` on zeroed arrays: signal phase `(row + col) % 2`, every entry gap the full link. */
extern "C" __global__ void init_state(const DeviceParams p, NodeMeta *meta, unsigned char *gaps)
{
    const unsigned int node = this_node(p);
    if (node == p.nodes) {
        return;
    }
    meta[node].phase = (unsigned char)((node / p.cols + node % p.cols) % 2);
    for (unsigned int d = 0; d < 4; ++d) {
        gaps[link_id(node, d)] = (unsigned char)p.link_cells;
    }
}

/** Mirror of `City::harvest_sensors`: copies each node's sensor count to `counts` and resets it. */
extern "C" __global__ void harvest(const DeviceParams p, NodeMeta *meta, unsigned int *counts)
{
    const unsigned int node = this_node(p);
    if (node == p.nodes) {
        return;
    }
    counts[node] = meta[node].sensor;
    meta[node].sensor = 0;
}

/** Sum of `v` over the 32 lanes of a warp, in lane 0. */
__device__ unsigned long long warp_sum(unsigned long long v)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

/** Mirror of `invariants::census`: adds spawned, exited, parked, crossed, active and stopped to `totals[0..6]`.
 * Launched with whole warps. */
extern "C" __global__ void census(const DeviceParams p, const Car *cars, const NodeMeta *meta, unsigned long long *totals)
{
    const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long values[6] = {0, 0, 0, 0, 0, 0};
    if (node < p.nodes) {
        const unsigned int cells = p.link_cells;
        const NodeMeta m = meta[node];
        values[0] = m.spawned;
        values[1] = m.exited;
        values[2] = m.parked;
        values[3] = m.crossed;
        for (unsigned int d = 0; d < 4; ++d) {
            const Car *link = cars + ((unsigned long long)node * 4 + d) * cells;
            values[4] += m.len[d];
            for (unsigned int i = 0; i < m.len[d]; ++i) {
                values[5] += link[slot(m.start[d], i, cells)].vel == 0;
            }
        }
    }
    for (unsigned int k = 0; k < 6; ++k) {
        const unsigned long long sum = warp_sum(values[k]);
        if ((threadIdx.x & 31) == 0) {
            atomicAdd(&totals[k], sum);
        }
    }
}

/** Mirror of `checksum::state_checksum`: adds every node's term to `*sum`, wrapping. Launched with whole warps. */
extern "C" __global__ void state_checksum(const DeviceParams p, const Car *cars, const NodeMeta *meta,
                                          unsigned long long *sum)
{
    const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long term = 0;
    if (node < p.nodes) {
        const Car *node_cars = cars + (unsigned long long)node * 4 * p.link_cells;
        term = splitmix64(node_hash(meta[node], node_cars, p.link_cells) + (unsigned long long)node * GOLDEN_GAMMA);
    }
    term = warp_sum(term);
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(sum, term);
    }
}

/** Philox of `n` independent counter and key pairs, one thread each. Used by the self-test. */
extern "C" __global__ void philox_batch(const uint4 *ctr, const uint2 *key, uint4 *out, unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = philox4x32_10(ctr[i], key[i]);
    }
}

/** A kernel that does nothing, for measuring what one launch costs. */
extern "C" __global__ void empty()
{
}
