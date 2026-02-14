#include <GLUT/glut.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

struct Bot {
    float x;
    float y;
    float heading;     // radians in world XZ plane
    bool hasLLD;
    int city;
    int layer;
    float layerPos; // supports smooth layer transitions on ramps
    bool onRamp;
    int rampFromLayer;
    int rampToLayer;
    float rampProgress;
    bool onSkyway;
    int skywayFromCity;
    int skywayToCity;
    float skywayProgress;
    float skywayStartX;
    float skywayStartY;
    float skywayEndX;
    float skywayEndY;
    float stateTimer;
    float stallTimer;
    float wobblePhase; // per-bot phase offset
    float wobbleFreq;  // radians/sec
    float limpSeverity;
    float leftLegPower;
    float rightLegPower;
    float leftTurnBias;
    float speedMultiplier;
    float wanderPhase;
    float wanderFreq;
    float wanderStrength;
    float turnMemory;
    struct TrailPoint {
        float x;
        float y;
        float layerPos;
    };
    std::vector<TrailPoint> trail;
};

namespace {
constexpr float kWorldDiameter = 110000.0f;
constexpr float kWorldRadius = kWorldDiameter * 0.5f;
constexpr int kGreenBotCount = 2000;
constexpr int kOrangeBotCount = 2000;
constexpr float kGoalRespawnSeconds = 60.0f;
constexpr float kOrangeSpeedMultiplier = 1.10f;
constexpr int kCityGridSize = 9;
constexpr int kCityCount = kCityGridSize * kCityGridSize;
constexpr float kCitySpacing = 10500.0f;
constexpr int kMinCityLayers = 10;
constexpr int kMaxCityLayers = 90;
constexpr float kLayerSpacing = 650.0f;
constexpr float kRampInnerRadius = 130.0f;
constexpr float kRampOuterRadius = 250.0f;
constexpr float kRampCenterRadius = 190.0f;
constexpr float kRampTravelSeconds = 5.4f;

// Movement profile to emulate leg length discrepancy.
constexpr float kStandardSpeed = 120.0f;        // px/s reference
constexpr float kBotSpeedFactor = 0.72f;        // slower than standard
constexpr float kEfficiencyLoss = 0.72f;        // less efficient movement
constexpr float kTurnRateLeft = 3.0f;           // rad/s (easier left turn)
constexpr float kTurnRateRight = 0.95f;         // rad/s (harder right turn)
constexpr float kWobbleAmplitude = 0.42f;       // rad
constexpr float kWobbleVelocityPenalty = 0.40f; // speed reduction from wobble
constexpr float kBotBodyLength = 68.0f;         // visual + boundary buffer
constexpr float kTrailMinPointDistance = 18.0f;
constexpr float kAlwaysLeftPull = 0.24f;        // persistent left drift from discrepancy

constexpr float kGroundY = 0.0f;
constexpr float kTrailY = 2.0f;
constexpr float kBotBaseY = 18.0f;
constexpr float kWallHeight = 200.0f;
constexpr float kTerrainAmplitude = 56.0f;
constexpr float kTerrainFrequencyA = 0.00125f;
constexpr float kTerrainFrequencyB = 0.0019f;
constexpr float kTerrainFrequencyC = 0.0026f;
constexpr int kTerrainRings = 72;
constexpr int kTerrainSegments = 220;

constexpr std::array<std::array<float, 2>, 16> kBuildingOffsets = {{
    {{-1900.0f, -1900.0f}}, {{-650.0f, -1900.0f}},  {{650.0f, -1900.0f}},  {{1900.0f, -1900.0f}},
    {{-1900.0f, -650.0f}},  {{1900.0f, -650.0f}},   {{-1900.0f, 650.0f}},  {{1900.0f, 650.0f}},
    {{-1900.0f, 1900.0f}},  {{-650.0f, 1900.0f}},   {{650.0f, 1900.0f}},   {{1900.0f, 1900.0f}},
    {{-2900.0f, -400.0f}},  {{2900.0f, 400.0f}},    {{-400.0f, -2900.0f}}, {{400.0f, 2900.0f}},
}};

GLuint groundTexture = 0;
GLuint worldDisplayList = 0;
GLuint botHighDisplayList = 0;
GLuint botLowDisplayList = 0;
GLuint botHighOrangeDisplayList = 0;
GLuint botLowOrangeDisplayList = 0;

std::mt19937 rng;
std::uniform_real_distribution<float> unit01(0.0f, 1.0f);
std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * static_cast<float>(M_PI));
std::uniform_real_distribution<float> noiseDist(-1.0f, 1.0f);

std::vector<Bot> bots;
std::array<int, kCityCount> cityLayerCounts{};
int maxCityLayers = kMinCityLayers;
std::vector<std::array<float, 2>> cityCenters;
std::vector<std::array<int, 2>> skywayEdges;

float goalX = 0.0f;
float goalY = 0.0f;
int goalLayer = 0;
int goalCity = 0;
float lastGoalSpawnTime = 0.0f;
float previousTime = 0.0f;
float currentSimTime = 0.0f;
float cameraDistance = 9800.0f;
bool zoomDragActive = false;
int lastMouseY = 0;
int zoomDragButton = -1;
int windowWidth = 1280;
int windowHeight = 900;
double fpsValue = 0.0;
int fpsFrameCount = 0;
float fpsAccumTime = 0.0f;
float cameraPosX = 0.0f;
float cameraPosY = 0.0f;
float cameraPosZ = 0.0f;
float cameraTargetX = 0.0f;
float cameraTargetY = 0.0f;
float cameraTargetZ = 0.0f;
int cameraTargetCity = 0;
bool exportRenderPass = false;
float timeScale = 1.0f;

constexpr float kCameraTiltRatio = 0.19f;
constexpr float kMinCameraDistance = 700.0f;
constexpr float kMaxCameraDistance = 170000.0f;
constexpr float kTargetFps = 30.0f;
constexpr float kFrameInterval = 1.0f / kTargetFps;
float nextFrameTime = 0.0f;

enum class CameraViewMode { TopDown = 1, Isometric = 2, Side = 3 };
CameraViewMode cameraViewMode = CameraViewMode::TopDown;

void setProjection(int w, int h) {
    const float aspect = (h == 0) ? 1.0f : static_cast<float>(w) / static_cast<float>(h);
    const float nearPlane = std::max(2.5f, cameraDistance * 0.0035f);
    const float farPlane = std::max(100000.0f, cameraDistance + 100000.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(52.0, aspect, nearPlane, farPlane);
    glMatrixMode(GL_MODELVIEW);
}

void setProjectionTiled(int fullW, int fullH, int tileX, int tileY, int tileW, int tileH) {
    const float nearPlane = std::max(2.5f, cameraDistance * 0.0035f);
    const float farPlane = std::max(100000.0f, cameraDistance + 100000.0f);
    const float aspect = static_cast<float>(fullW) / static_cast<float>(fullH);
    const float fovyRad = 52.0f * static_cast<float>(M_PI) / 180.0f;
    const float top = nearPlane * std::tan(0.5f * fovyRad);
    const float right = top * aspect;

    const float leftN = -right + (2.0f * right) * (static_cast<float>(tileX) / static_cast<float>(fullW));
    const float rightN =
        -right + (2.0f * right) * (static_cast<float>(tileX + tileW) / static_cast<float>(fullW));
    const float bottomN = -top + (2.0f * top) * (static_cast<float>(tileY) / static_cast<float>(fullH));
    const float topN = -top + (2.0f * top) * (static_cast<float>(tileY + tileH) / static_cast<float>(fullH));

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(leftN, rightN, bottomN, topN, nearPlane, farPlane);
    glMatrixMode(GL_MODELVIEW);
}

uint32_t crc32UpdateRaw(uint32_t crc, const unsigned char* data, size_t len) {
    static uint32_t table[256];
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                c = (c & 1U) ? (0xEDB88320U ^ (c >> 1U)) : (c >> 1U);
            }
            table[i] = c;
        }
        initialized = true;
    }
    for (size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFU] ^ (crc >> 8U);
    }
    return crc;
}

void writeU32BE(std::ofstream& out, uint32_t v) {
    unsigned char b[4] = {
        static_cast<unsigned char>((v >> 24U) & 0xFFU),
        static_cast<unsigned char>((v >> 16U) & 0xFFU),
        static_cast<unsigned char>((v >> 8U) & 0xFFU),
        static_cast<unsigned char>(v & 0xFFU),
    };
    out.write(reinterpret_cast<const char*>(b), 4);
}

void beginChunk(std::ofstream& out, const char type[4], uint32_t len, uint32_t& crc) {
    writeU32BE(out, len);
    out.write(type, 4);
    crc = 0xFFFFFFFFU;
    crc = crc32UpdateRaw(crc, reinterpret_cast<const unsigned char*>(type), 4);
}

void writeChunkData(std::ofstream& out, uint32_t& crc, const unsigned char* data, size_t len) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(len));
    crc = crc32UpdateRaw(crc, data, len);
}

void endChunk(std::ofstream& out, uint32_t crc) {
    writeU32BE(out, crc ^ 0xFFFFFFFFU);
}

void writeChunk(std::ofstream& out, const char type[4], const unsigned char* data, uint32_t len) {
    uint32_t crc = 0;
    beginChunk(out, type, len, crc);
    if (len > 0) {
        writeChunkData(out, crc, data, len);
    }
    endChunk(out, crc);
}

uint32_t adler32Update(uint32_t adler, const unsigned char* data, size_t len) {
    uint32_t s1 = adler & 0xFFFFU;
    uint32_t s2 = (adler >> 16U) & 0xFFFFU;
    constexpr uint32_t mod = 65521U;
    for (size_t i = 0; i < len; ++i) {
        s1 += static_cast<uint32_t>(data[i]);
        if (s1 >= mod) s1 -= mod;
        s2 += s1;
        if (s2 >= mod) s2 -= mod;
    }
    return (s2 << 16U) | s1;
}

bool writePngRgb(const std::string& path, int width, int height, const std::vector<unsigned char>& rgbTopDown) {
    if (width <= 0 || height <= 0) return false;
    const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height) * 3U;
    if (rgbTopDown.size() != expected) return false;

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    const unsigned char sig[8] = {137U, 80U, 78U, 71U, 13U, 10U, 26U, 10U};
    out.write(reinterpret_cast<const char*>(sig), 8);

    unsigned char ihdr[13] = {};
    ihdr[0] = static_cast<unsigned char>((static_cast<uint32_t>(width) >> 24U) & 0xFFU);
    ihdr[1] = static_cast<unsigned char>((static_cast<uint32_t>(width) >> 16U) & 0xFFU);
    ihdr[2] = static_cast<unsigned char>((static_cast<uint32_t>(width) >> 8U) & 0xFFU);
    ihdr[3] = static_cast<unsigned char>(static_cast<uint32_t>(width) & 0xFFU);
    ihdr[4] = static_cast<unsigned char>((static_cast<uint32_t>(height) >> 24U) & 0xFFU);
    ihdr[5] = static_cast<unsigned char>((static_cast<uint32_t>(height) >> 16U) & 0xFFU);
    ihdr[6] = static_cast<unsigned char>((static_cast<uint32_t>(height) >> 8U) & 0xFFU);
    ihdr[7] = static_cast<unsigned char>(static_cast<uint32_t>(height) & 0xFFU);
    ihdr[8] = 8U;  // bit depth
    ihdr[9] = 2U;  // RGB
    ihdr[10] = 0U; // compression
    ihdr[11] = 0U; // filter
    ihdr[12] = 0U; // interlace
    writeChunk(out, "IHDR", ihdr, 13U);

    std::vector<unsigned char> zbuf;
    zbuf.reserve(1U << 20U);
    auto flushIdat = [&]() {
        if (!zbuf.empty()) {
            writeChunk(out, "IDAT", zbuf.data(), static_cast<uint32_t>(zbuf.size()));
            zbuf.clear();
        }
    };
    auto emitByte = [&](unsigned char b) {
        zbuf.push_back(b);
        if (zbuf.size() >= (1U << 20U)) flushIdat();
    };

    emitByte(0x78U);
    emitByte(0x01U);

    uint32_t adler = 1U;
    const size_t rowBytes = static_cast<size_t>(width) * 3U;
    const uint64_t totalBytes = static_cast<uint64_t>(height) * static_cast<uint64_t>(rowBytes + 1U);
    uint64_t written = 0;
    int row = 0;
    size_t rowPos = 0;

    while (written < totalBytes) {
        const uint32_t blockLen = static_cast<uint32_t>(std::min<uint64_t>(65535ULL, totalBytes - written));
        const unsigned char bfinal = (written + static_cast<uint64_t>(blockLen) == totalBytes) ? 1U : 0U;
        emitByte(bfinal);
        emitByte(static_cast<unsigned char>(blockLen & 0xFFU));
        emitByte(static_cast<unsigned char>((blockLen >> 8U) & 0xFFU));
        const uint16_t nlen = static_cast<uint16_t>(~blockLen);
        emitByte(static_cast<unsigned char>(nlen & 0xFFU));
        emitByte(static_cast<unsigned char>((nlen >> 8U) & 0xFFU));

        std::vector<unsigned char> block;
        block.reserve(blockLen);
        for (uint32_t i = 0; i < blockLen; ++i) {
            if (rowPos == 0) {
                block.push_back(0U);
                rowPos = 1;
            } else {
                const size_t src = static_cast<size_t>(row) * rowBytes + (rowPos - 1U);
                block.push_back(rgbTopDown[src]);
                ++rowPos;
                if (rowPos > rowBytes) {
                    rowPos = 0;
                    ++row;
                }
            }
        }
        adler = adler32Update(adler, block.data(), block.size());
        for (unsigned char v : block) emitByte(v);
        written += static_cast<uint64_t>(blockLen);
    }

    emitByte(static_cast<unsigned char>((adler >> 24U) & 0xFFU));
    emitByte(static_cast<unsigned char>((adler >> 16U) & 0xFFU));
    emitByte(static_cast<unsigned char>((adler >> 8U) & 0xFFU));
    emitByte(static_cast<unsigned char>(adler & 0xFFU));
    flushIdat();

    writeChunk(out, "IEND", nullptr, 0U);
    const bool writeOk = out.good();
    out.close();
    if (!writeOk) return false;

    // Re-encode with system PNG encoder when available for much smaller files.
    const std::string optimizedPath = path + ".opt.png";
    const std::string cmd =
        "sips -s format png \"" + path + "\" --out \"" + optimizedPath + "\" >/dev/null 2>&1";
    if (std::system(cmd.c_str()) == 0) {
        std::remove(path.c_str());
        if (std::rename(optimizedPath.c_str(), path.c_str()) != 0) {
            // If rename fails, keep original and remove temp.
            std::remove(optimizedPath.c_str());
        }
    } else {
        std::remove(optimizedPath.c_str());
    }
    return true;
}

float clampAbs(float value, float maxAbs) {
    if (value > maxAbs) return maxAbs;
    if (value < -maxAbs) return -maxAbs;
    return value;
}

float terrainHeight(float x, float z) {
    // Brutalist city floors are largely flat.
    const float subtle = 2.0f * std::sin(x * 0.0007f) * std::cos(z * 0.0006f);
    return kGroundY + subtle;
}

float layerBaseHeight(float layerPos) {
    return layerPos * kLayerSpacing;
}

int cityIndexForPosition(float x, float y) {
    int best = 0;
    float bestD2 = 1.0e30f;
    for (int i = 0; i < kCityCount; ++i) {
        const float dx = x - cityCenters[i][0];
        const float dy = y - cityCenters[i][1];
        const float d2 = dx * dx + dy * dy;
        if (d2 < bestD2) {
            bestD2 = d2;
            best = i;
        }
    }
    return best;
}

void initCityLayout() {
    cityCenters.clear();
    cityCenters.reserve(kCityCount);
    const float offset = 0.5f * static_cast<float>(kCityGridSize - 1);
    for (int r = 0; r < kCityGridSize; ++r) {
        for (int c = 0; c < kCityGridSize; ++c) {
            cityCenters.push_back(
                {(static_cast<float>(c) - offset) * kCitySpacing, (static_cast<float>(r) - offset) * kCitySpacing});
        }
    }

    // Sparse but fully connected skyway graph: a single serpentine chain through all cities.
    skywayEdges.clear();
    skywayEdges.reserve(kCityCount - 1);
    std::vector<int> order;
    order.reserve(kCityCount);
    for (int r = 0; r < kCityGridSize; ++r) {
        if ((r % 2) == 0) {
            for (int c = 0; c < kCityGridSize; ++c) order.push_back(r * kCityGridSize + c);
        } else {
            for (int c = kCityGridSize - 1; c >= 0; --c) order.push_back(r * kCityGridSize + c);
        }
    }
    for (size_t i = 1; i < order.size(); ++i) {
        skywayEdges.push_back({order[i - 1], order[i]});
    }
}

void initCityLayers() {
    std::uniform_int_distribution<int> layerDist(kMinCityLayers, kMaxCityLayers);
    maxCityLayers = kMinCityLayers;
    for (int i = 0; i < kCityCount; ++i) {
        cityLayerCounts[i] = layerDist(rng);
        if (cityLayerCounts[i] > maxCityLayers) maxCityLayers = cityLayerCounts[i];
    }
}

int nextCityOnPath(int startCity, int goalCity) {
    if (startCity == goalCity) return startCity;
    std::array<int, kCityCount> prev{};
    std::array<bool, kCityCount> visited{};
    for (int i = 0; i < kCityCount; ++i) prev[i] = -1;
    std::array<int, kCityCount> q{};
    int head = 0;
    int tail = 0;
    q[tail++] = startCity;
    visited[startCity] = true;

    while (head < tail) {
        const int u = q[head++];
        if (u == goalCity) break;
        for (const auto& e : skywayEdges) {
            int v = -1;
            if (e[0] == u) v = e[1];
            if (e[1] == u) v = e[0];
            if (v >= 0 && !visited[v]) {
                visited[v] = true;
                prev[v] = u;
                q[tail++] = v;
            }
        }
    }

    if (!visited[goalCity]) return startCity;
    int cur = goalCity;
    while (prev[cur] >= 0 && prev[cur] != startCity) {
        cur = prev[cur];
    }
    return (prev[cur] == startCity) ? cur : goalCity;
}

void skywayEndpoints(int fromCity, int toCity, float& sx, float& sy, float& ex, float& ey) {
    const float fx = cityCenters[fromCity][0];
    const float fy = cityCenters[fromCity][1];
    const float tx = cityCenters[toCity][0];
    const float ty = cityCenters[toCity][1];
    const float dx = tx - fx;
    const float dy = ty - fy;
    const float len = std::sqrt(dx * dx + dy * dy);
    const float nx = (len > 0.0f) ? (dx / len) : 1.0f;
    const float ny = (len > 0.0f) ? (dy / len) : 0.0f;
    constexpr float edgeOffset = 3600.0f;
    sx = fx + nx * edgeOffset;
    sy = fy + ny * edgeOffset;
    ex = tx - nx * edgeOffset;
    ey = ty - ny * edgeOffset;
}

void nearestCityCenter(float x, float y, float& outX, float& outY) {
    float bestD2 = 1.0e30f;
    outX = 0.0f;
    outY = 0.0f;
    for (const auto& c : cityCenters) {
        const float dx = x - c[0];
        const float dy = y - c[1];
        const float d2 = dx * dx + dy * dy;
        if (d2 < bestD2) {
            bestD2 = d2;
            outX = c[0];
            outY = c[1];
        }
    }
}

float isometricFitDistance() {
    const float stackHeight = layerBaseHeight(static_cast<float>(maxCityLayers - 1));
    const float citySpan = static_cast<float>(kCityGridSize - 1) * kCitySpacing + 8600.0f;
    const float baseFit = std::max(citySpan * 1.2f, stackHeight * 1.6f);
    return baseFit * 1.10f; // extra safety margin to avoid edge clipping in isometric framing
}

float sideFitDistance() {
    const float stackHeight = layerBaseHeight(static_cast<float>(maxCityLayers - 1));
    const float citySpan = static_cast<float>(kCityGridSize - 1) * kCitySpacing + 8600.0f;
    return std::max(citySpan * 1.35f, stackHeight * 2.90f);
}

void terrainGradient(float x, float z, float& dhdx, float& dhdz) {
    constexpr float e = 3.0f;
    const float hL = terrainHeight(x - e, z);
    const float hR = terrainHeight(x + e, z);
    const float hD = terrainHeight(x, z - e);
    const float hU = terrainHeight(x, z + e);
    dhdx = (hR - hL) / (2.0f * e);
    dhdz = (hU - hD) / (2.0f * e);
}

void terrainNormal(float x, float z, float& nx, float& ny, float& nz) {
    float dhdx = 0.0f;
    float dhdz = 0.0f;
    terrainGradient(x, z, dhdx, dhdz);
    nx = -dhdx;
    ny = 1.0f;
    nz = -dhdz;
    const float invLen = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
    nx *= invLen;
    ny *= invLen;
    nz *= invLen;
}

float terrainRoughness(float x, float z) {
    float dhdx = 0.0f;
    float dhdz = 0.0f;
    terrainGradient(x, z, dhdx, dhdz);
    return std::sqrt(dhdx * dhdx + dhdz * dhdz);
}

void createGroundTexture() {
    constexpr int texSize = 256;
    std::vector<unsigned char> tex(texSize * texSize * 3);
    for (int y = 0; y < texSize; ++y) {
        for (int x = 0; x < texSize; ++x) {
            const float fx = static_cast<float>(x) / static_cast<float>(texSize);
            const float fy = static_cast<float>(y) / static_cast<float>(texSize);
            const float p1 = 0.5f + 0.5f * std::sin(42.0f * fx + 0.9f) * std::cos(39.0f * fy - 0.7f);
            const float p2 = 0.5f + 0.5f * std::sin(140.0f * fx) * std::sin(130.0f * fy);
            const float p3 = 0.5f + 0.5f * std::sin(70.0f * (fx - fy) + 0.5f);
            const float blend = 0.55f * p1 + 0.30f * p2 + 0.15f * p3;

            float rf = 0.26f + 0.32f * blend;
            float gf = 0.27f + 0.33f * blend;
            float bf = 0.29f + 0.34f * blend;

            const unsigned char r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, rf * 255.0f)));
            const unsigned char g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, gf * 255.0f)));
            const unsigned char b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, bf * 255.0f)));
            const int i = (y * texSize + x) * 3;
            tex[i + 0] = r;
            tex[i + 1] = g;
            tex[i + 2] = b;
        }
    }

    glGenTextures(1, &groundTexture);
    glBindTexture(GL_TEXTURE_2D, groundTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, texSize, texSize, GL_RGB, GL_UNSIGNED_BYTE, tex.data());
}

float wrapAngle(float a) {
    while (a > static_cast<float>(M_PI)) a -= 2.0f * static_cast<float>(M_PI);
    while (a < -static_cast<float>(M_PI)) a += 2.0f * static_cast<float>(M_PI);
    return a;
}

void applyZoom(float delta) {
    cameraDistance = std::max(kMinCameraDistance, std::min(kMaxCameraDistance, cameraDistance + delta));
    glutPostRedisplay();
}

void spawnGoal() {
    const int city = static_cast<int>(unit01(rng) * static_cast<float>(kCityCount));
    goalCity = (city >= kCityCount) ? (kCityCount - 1) : city;
    const int b = static_cast<int>(unit01(rng) * static_cast<float>(kBuildingOffsets.size()));
    const int bIdx = (b >= static_cast<int>(kBuildingOffsets.size())) ? (static_cast<int>(kBuildingOffsets.size()) - 1) : b;
    goalX = cityCenters[goalCity][0] + kBuildingOffsets[bIdx][0];
    goalY = cityCenters[goalCity][1] + kBuildingOffsets[bIdx][1];
    goalLayer = static_cast<int>(unit01(rng) * static_cast<float>(cityLayerCounts[goalCity]));
    if (goalLayer >= cityLayerCounts[goalCity]) goalLayer = cityLayerCounts[goalCity] - 1;
}

void initBots() {
    bots.clear();
    bots.reserve(kGreenBotCount + kOrangeBotCount);

    for (int i = 0; i < kGreenBotCount; ++i) {
        const int city = static_cast<int>(unit01(rng) * static_cast<float>(kCityCount));
        const int cityIdx = (city >= kCityCount) ? (kCityCount - 1) : city;
        const float r = 3000.0f * std::sqrt(unit01(rng));
        const float t = angleDist(rng);

        Bot b;
        b.x = cityCenters[cityIdx][0] + r * std::cos(t);
        b.y = cityCenters[cityIdx][1] + r * std::sin(t);
        b.hasLLD = true;
        b.city = cityIdx;
        b.layer = static_cast<int>(unit01(rng) * static_cast<float>(cityLayerCounts[cityIdx]));
        if (b.layer >= cityLayerCounts[cityIdx]) b.layer = cityLayerCounts[cityIdx] - 1;
        b.layerPos = static_cast<float>(b.layer);
        b.onRamp = false;
        b.rampFromLayer = b.layer;
        b.rampToLayer = b.layer;
        b.rampProgress = 0.0f;
        b.onSkyway = false;
        b.skywayFromCity = b.city;
        b.skywayToCity = b.city;
        b.skywayProgress = 0.0f;
        b.skywayStartX = b.x;
        b.skywayStartY = b.y;
        b.skywayEndX = b.x;
        b.skywayEndY = b.y;
        b.stateTimer = 0.0f;
        b.stallTimer = 0.0f;
        b.heading = angleDist(rng);
        b.wobblePhase = angleDist(rng);
        b.wobbleFreq = 2.0f + 3.0f * unit01(rng);
        b.limpSeverity = 0.55f + 0.4f * unit01(rng);
        b.leftLegPower = 0.60f + 0.18f * unit01(rng);
        b.rightLegPower = 0.95f + 0.12f * unit01(rng);
        b.leftTurnBias = 0.10f + 0.28f * unit01(rng);
        b.speedMultiplier = 1.0f;
        b.wanderPhase = angleDist(rng);
        b.wanderFreq = 0.35f + 0.95f * unit01(rng);
        b.wanderStrength = 0.16f + 0.20f * unit01(rng);
        b.turnMemory = 0.0f;
        b.trail.reserve(2048);
        b.trail.push_back({b.x, b.y, b.layerPos});
        bots.push_back(b);
    }

    for (int i = 0; i < kOrangeBotCount; ++i) {
        const int city = static_cast<int>(unit01(rng) * static_cast<float>(kCityCount));
        const int cityIdx = (city >= kCityCount) ? (kCityCount - 1) : city;
        const float r = 3000.0f * std::sqrt(unit01(rng));
        const float t = angleDist(rng);

        Bot b;
        b.x = cityCenters[cityIdx][0] + r * std::cos(t);
        b.y = cityCenters[cityIdx][1] + r * std::sin(t);
        b.hasLLD = false;
        b.city = cityIdx;
        b.layer = static_cast<int>(unit01(rng) * static_cast<float>(cityLayerCounts[cityIdx]));
        if (b.layer >= cityLayerCounts[cityIdx]) b.layer = cityLayerCounts[cityIdx] - 1;
        b.layerPos = static_cast<float>(b.layer);
        b.onRamp = false;
        b.rampFromLayer = b.layer;
        b.rampToLayer = b.layer;
        b.rampProgress = 0.0f;
        b.onSkyway = false;
        b.skywayFromCity = b.city;
        b.skywayToCity = b.city;
        b.skywayProgress = 0.0f;
        b.skywayStartX = b.x;
        b.skywayStartY = b.y;
        b.skywayEndX = b.x;
        b.skywayEndY = b.y;
        b.stateTimer = 0.0f;
        b.stallTimer = 0.0f;
        b.heading = angleDist(rng);
        b.wobblePhase = angleDist(rng);
        b.wobbleFreq = 1.8f + 2.2f * unit01(rng);
        b.limpSeverity = 0.0f;
        b.leftLegPower = 1.0f;
        b.rightLegPower = 1.0f;
        b.leftTurnBias = 0.0f;
        b.speedMultiplier = kOrangeSpeedMultiplier;
        b.wanderPhase = angleDist(rng);
        b.wanderFreq = 0.28f + 0.55f * unit01(rng);
        b.wanderStrength = 0.06f + 0.08f * unit01(rng);
        b.turnMemory = 0.0f;
        b.trail.reserve(2048);
        b.trail.push_back({b.x, b.y, b.layerPos});
        bots.push_back(b);
    }
}

void drawCityBox(float x, float y, float z, float sx, float sy, float sz, float r, float g, float b) {
    glColor3f(r, g, b);
    glPushMatrix();
    glTranslatef(x, y, z);
    glScalef(sx, sy, sz);
    glutSolidCube(1.0f);
    glPopMatrix();
}

void drawCityWalkway(float ax, float az, float bx, float bz, float y, float thickness) {
    const float dx = bx - ax;
    const float dz = bz - az;
    const float len = std::sqrt(dx * dx + dz * dz);
    if (len < 1.0f) return;
    const float mx = 0.5f * (ax + bx);
    const float mz = 0.5f * (az + bz);
    const float angDeg = std::atan2(dz, dx) * 57.2957795f;

    glPushMatrix();
    glTranslatef(mx, y, mz);
    glRotatef(-angDeg, 0.0f, 1.0f, 0.0f);
    glScalef(len, thickness, 80.0f);
    glutSolidCube(1.0f);
    glPopMatrix();
}

void drawWorldGeometry() {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, groundTexture);

    const int localWalkwayPairs[][2] = {
        {0, 1}, {2, 3}, {8, 9}, {10, 11}, {0, 4}, {3, 5}, {8, 6}, {11, 7}, {1, 4}, {2, 5}, {12, 4}, {13, 5},
    };

    const bool topMode = (cameraViewMode == CameraViewMode::TopDown);
    const int layerStep = exportRenderPass ? 1 : (topMode ? 1 : ((cameraDistance > 70000.0f) ? 4 : ((cameraDistance > 45000.0f) ? 3 : 2)));
    const int buildingStep =
        exportRenderPass ? 1 : (topMode ? 1 : ((cameraDistance > 65000.0f) ? 3 : ((cameraDistance > 42000.0f) ? 2 : 1)));
    const int walkwayStep = exportRenderPass ? 1 : (topMode ? 1 : ((cameraDistance > 52000.0f) ? 2 : 1));

    for (int city = 0; city < kCityCount; ++city) {
        const float cx = cityCenters[city][0];
        const float cy = cityCenters[city][1];
        const float dxCam = cx - cameraTargetX;
        const float dyCam = cy - cameraTargetZ;
        const float cityDist2 = dxCam * dxCam + dyCam * dyCam;
        if (!exportRenderPass && topMode && cityDist2 > 22000.0f * 22000.0f) continue;

        for (int layer = 0; layer < cityLayerCounts[city]; layer += layerStep) {
            const float baseY = layerBaseHeight(static_cast<float>(layer));
            if (!exportRenderPass && topMode && std::fabs(baseY - cameraTargetY) > 6.0f * kLayerSpacing) continue;

            // City platform slab.
            glColor3f(0.36f, 0.38f, 0.41f);
            glBegin(GL_QUADS);
            glNormal3f(0.0f, 1.0f, 0.0f);
            glTexCoord2f(-3.0f, -3.0f);
            glVertex3f(cx - 4200.0f, baseY, cy - 4200.0f);
            glTexCoord2f(3.0f, -3.0f);
            glVertex3f(cx + 4200.0f, baseY, cy - 4200.0f);
            glTexCoord2f(3.0f, 3.0f);
            glVertex3f(cx + 4200.0f, baseY, cy + 4200.0f);
            glTexCoord2f(-3.0f, 3.0f);
            glVertex3f(cx - 4200.0f, baseY, cy + 4200.0f);
            glEnd();
            drawCityBox(cx, baseY - 18.0f, cy, 8500.0f, 36.0f, 8500.0f, 0.20f, 0.21f, 0.23f);

            // Buildings from stacked basic shapes.
            for (int i = 0; i < 16; i += buildingStep) {
                const float bx = cx + kBuildingOffsets[i][0];
                const float bz = cy + kBuildingOffsets[i][1];
                const int tiers = 2 + ((i + layer + city) % 3);
                float width = 740.0f - 35.0f * static_cast<float>(i % 3);
                float depth = 700.0f - 40.0f * static_cast<float>((i + 1) % 3);
                float yCursor = baseY + 90.0f;
                for (int t = 0; t < tiers; ++t) {
                    const float h = 190.0f + 40.0f * static_cast<float>((t + i + city) % 2);
                    drawCityBox(bx, yCursor + h * 0.5f, bz, width, h, depth, 0.30f - 0.02f * t, 0.31f - 0.02f * t,
                                0.34f - 0.02f * t);
                    yCursor += h;
                    width *= 0.82f;
                    depth *= 0.82f;
                }
            }

            // Sparse local walkways.
            glColor3f(0.48f, 0.50f, 0.54f);
            for (int wi = 0; wi < static_cast<int>(sizeof(localWalkwayPairs) / sizeof(localWalkwayPairs[0])); wi += walkwayStep) {
                const auto& pair = localWalkwayPairs[wi];
                const int a = pair[0];
                const int b = pair[1];
                drawCityWalkway(cx + kBuildingOffsets[a][0], cy + kBuildingOffsets[a][1], cx + kBuildingOffsets[b][0],
                                cy + kBuildingOffsets[b][1],
                                baseY + 230.0f, 22.0f);
            }

            // Core and ramp connectors for this city.
            drawCityBox(cx, baseY + 160.0f, cy, 560.0f, 320.0f, 560.0f, 0.40f, 0.42f, 0.45f);
            drawCityWalkway(cx, cy, cx + kRampCenterRadius, cy, baseY + 58.0f, 18.0f);
            drawCityWalkway(cx, cy, cx - kRampCenterRadius, cy, baseY + 58.0f, 18.0f);
            drawCityWalkway(cx, cy, cx, cy + kRampCenterRadius, baseY + 58.0f, 18.0f);
            drawCityWalkway(cx, cy, cx, cy - kRampCenterRadius, baseY + 58.0f, 18.0f);
        }
    }

    // Sparse inter-city skyways per shared layer.
    glColor3f(0.80f, 0.82f, 0.88f);
    for (const auto& e : skywayEdges) {
        const int a = e[0];
        const int b = e[1];
        const int sharedLayers = std::min(cityLayerCounts[a], cityLayerCounts[b]);
        for (int layer = 0; layer < sharedLayers; layer += layerStep) {
            const float baseY = layerBaseHeight(static_cast<float>(layer));
            float sx, sy, ex, ey;
            skywayEndpoints(a, b, sx, sy, ex, ey);
            drawCityWalkway(sx, sy, ex, ey, baseY + 560.0f, 26.0f);
            glDisable(GL_LIGHTING);
            glColor4f(0.98f, 0.99f, 1.0f, 0.85f);
            glLineWidth(2.2f);
            glBegin(GL_LINES);
            glVertex3f(sx, baseY + 575.0f, sy);
            glVertex3f(ex, baseY + 575.0f, ey);
            glEnd();
            glEnable(GL_LIGHTING);
        }
    }
    glDisable(GL_TEXTURE_2D);

    // Ramp spines at each city center.
    for (int city = 0; city < kCityCount; ++city) {
        for (int layer = 0; layer < cityLayerCounts[city] - 1; ++layer) {
            const float y0 = layerBaseHeight(static_cast<float>(layer)) + 40.0f;
            const float y1 = layerBaseHeight(static_cast<float>(layer + 1)) + 40.0f;
            const float cx = cityCenters[city][0];
            const float cy = cityCenters[city][1];
            glBegin(GL_QUAD_STRIP);
            for (int i = 0; i <= 96; ++i) {
                const float t = static_cast<float>(i) / 96.0f;
                const float ang = 2.0f * static_cast<float>(M_PI) * t;
                const float c = std::cos(ang);
                const float s = std::sin(ang);
                const float y = y0 + (y1 - y0) * t;
                glNormal3f(c, 0.20f, s);
                glColor3f(0.44f, 0.46f, 0.49f);
                glVertex3f(cx + c * kRampInnerRadius, y, cy + s * kRampInnerRadius);
                glColor3f(0.76f, 0.78f, 0.82f);
                glVertex3f(cx + c * kRampOuterRadius, y, cy + s * kRampOuterRadius);
            }
            glEnd();
            glDisable(GL_LIGHTING);
            glColor4f(0.95f, 0.97f, 1.0f, 0.75f);
            glLineWidth(1.8f);
            glBegin(GL_LINE_STRIP);
            for (int i = 0; i <= 96; ++i) {
                const float t = static_cast<float>(i) / 96.0f;
                const float ang = 2.0f * static_cast<float>(M_PI) * t;
                const float c = std::cos(ang);
                const float s = std::sin(ang);
                const float y = y0 + (y1 - y0) * t;
                glVertex3f(cx + c * kRampCenterRadius, y + 3.0f, cy + s * kRampCenterRadius);
            }
            glEnd();
            glEnable(GL_LIGHTING);
        }
    }

    glDisable(GL_LIGHTING);
    glColor4f(0.90f, 0.92f, 0.95f, 0.32f);
    glLineWidth(1.8f);
    for (int city = 0; city < kCityCount; ++city) {
        for (int layer = 0; layer < cityLayerCounts[city]; ++layer) {
            const float baseY = layerBaseHeight(static_cast<float>(layer));
            const float cx = cityCenters[city][0];
            const float cy = cityCenters[city][1];
            glBegin(GL_LINE_LOOP);
            glVertex3f(cx - 4300.0f, baseY + 2.0f, cy - 4300.0f);
            glVertex3f(cx + 4300.0f, baseY + 2.0f, cy - 4300.0f);
            glVertex3f(cx + 4300.0f, baseY + 2.0f, cy + 4300.0f);
            glVertex3f(cx - 4300.0f, baseY + 2.0f, cy + 4300.0f);
            glEnd();
        }
    }
    glEnable(GL_LIGHTING);
}

void createDisplayLists() {
    if (botHighDisplayList == 0) botHighDisplayList = glGenLists(1);
    if (botLowDisplayList == 0) botLowDisplayList = glGenLists(1);
    if (botHighOrangeDisplayList == 0) botHighOrangeDisplayList = glGenLists(1);
    if (botLowOrangeDisplayList == 0) botLowOrangeDisplayList = glGenLists(1);

    glNewList(botHighDisplayList, GL_COMPILE);
    // Abstract rounded tortoise-like body.
    glColor3f(0.18f, 1.00f, 0.40f);
    glPushMatrix();
    glScalef(31.0f, 10.0f, 21.0f);
    glutSolidSphere(1.0f, 12, 10);
    glPopMatrix();

    glColor3f(0.62f, 1.00f, 0.70f);
    glPushMatrix();
    glTranslatef(4.0f, 7.8f, 0.0f);
    glScalef(20.0f, 6.0f, 13.0f);
    glutSolidSphere(1.0f, 12, 10);
    glPopMatrix();

    glColor3f(0.52f, 1.00f, 0.64f);
    glPushMatrix();
    glTranslatef(30.0f, 0.0f, 0.0f);
    glScalef(9.5f, 6.5f, 7.2f);
    glutSolidSphere(1.0f, 10, 8);
    glPopMatrix();

    // Clean line accents.
    glDisable(GL_LIGHTING);
    glColor3f(0.92f, 1.00f, 0.95f);
    glLineWidth(1.2f);
    glPushMatrix();
    glTranslatef(4.0f, 8.2f, 0.0f);
    glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
    glutWireTorus(0.7f, 13.8f, 10, 18);
    glPopMatrix();
    glBegin(GL_LINES);
    glVertex3f(-9.0f, 8.2f, 0.0f);
    glVertex3f(15.0f, 8.2f, 0.0f);
    glVertex3f(33.0f, 1.8f, -2.4f);
    glVertex3f(33.0f, 1.8f, 2.4f);
    glEnd();
    glEnable(GL_LIGHTING);
    glEndList();

    glNewList(botLowDisplayList, GL_COMPILE);
    glColor3f(0.22f, 1.00f, 0.42f);
    glPushMatrix();
    glScalef(26.0f, 8.0f, 17.0f);
    glutSolidSphere(1.0f, 9, 8);
    glPopMatrix();

    glColor3f(0.66f, 1.00f, 0.74f);
    glPushMatrix();
    glTranslatef(3.0f, 5.8f, 0.0f);
    glScalef(13.0f, 4.0f, 8.5f);
    glutSolidSphere(1.0f, 8, 6);
    glPopMatrix();
    glEndList();

    glNewList(botHighOrangeDisplayList, GL_COMPILE);
    glColor3f(1.00f, 0.52f, 0.08f);
    glPushMatrix();
    glScalef(31.0f, 10.0f, 21.0f);
    glutSolidSphere(1.0f, 12, 10);
    glPopMatrix();

    glColor3f(1.00f, 0.74f, 0.30f);
    glPushMatrix();
    glTranslatef(4.0f, 7.8f, 0.0f);
    glScalef(20.0f, 6.0f, 13.0f);
    glutSolidSphere(1.0f, 12, 10);
    glPopMatrix();

    glColor3f(1.00f, 0.78f, 0.34f);
    glPushMatrix();
    glTranslatef(30.0f, 0.0f, 0.0f);
    glScalef(9.5f, 6.5f, 7.2f);
    glutSolidSphere(1.0f, 10, 8);
    glPopMatrix();

    glDisable(GL_LIGHTING);
    glColor3f(1.00f, 0.93f, 0.82f);
    glLineWidth(1.2f);
    glPushMatrix();
    glTranslatef(4.0f, 8.2f, 0.0f);
    glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
    glutWireTorus(0.7f, 13.8f, 10, 18);
    glPopMatrix();
    glBegin(GL_LINES);
    glVertex3f(-9.0f, 8.2f, 0.0f);
    glVertex3f(15.0f, 8.2f, 0.0f);
    glVertex3f(33.0f, 1.8f, -2.4f);
    glVertex3f(33.0f, 1.8f, 2.4f);
    glEnd();
    glEnable(GL_LIGHTING);
    glEndList();

    glNewList(botLowOrangeDisplayList, GL_COMPILE);
    glColor3f(1.00f, 0.54f, 0.10f);
    glPushMatrix();
    glScalef(26.0f, 8.0f, 17.0f);
    glutSolidSphere(1.0f, 9, 8);
    glPopMatrix();

    glColor3f(1.00f, 0.74f, 0.30f);
    glPushMatrix();
    glTranslatef(3.0f, 5.8f, 0.0f);
    glScalef(13.0f, 4.0f, 8.5f);
    glutSolidSphere(1.0f, 8, 6);
    glPopMatrix();
    glEndList();
}

void drawWorld() {
    drawWorldGeometry();
}

void drawGoal() {
    glPushMatrix();
    glTranslatef(goalX, terrainHeight(goalX, goalY) + layerBaseHeight(static_cast<float>(goalLayer)) + 32.0f, goalY);

    glColor3f(0.98f, 0.28f, 0.20f);
    glPushMatrix();
    glScalef(1.0f, 1.6f, 1.0f);
    glutSolidSphere(44.0f, 14, 10);
    glPopMatrix();

    glColor3f(1.0f, 0.58f, 0.18f);
    glTranslatef(0.0f, 72.0f, 0.0f);
    glutSolidSphere(20.0f, 12, 10);

    glColor4f(1.0f, 0.45f, 0.18f, 0.30f);
    glDisable(GL_LIGHTING);
    glutWireSphere(88.0f, 16, 12);
    glEnable(GL_LIGHTING);

    glPopMatrix();
}

void drawLeg(float x, float z, float step, float lengthScale, float yBias) {
    glPushMatrix();
    glTranslatef(x + step, -9.0f + yBias + std::fabs(step) * 0.2f, z);
    glScalef(8.0f * lengthScale, 4.6f, 5.2f);
    glutSolidSphere(1.0f, 8, 7);
    glDisable(GL_LIGHTING);
    glColor3f(0.95f, 0.98f, 0.96f);
    glutWireSphere(1.05f, 8, 7);
    glEnable(GL_LIGHTING);
    glPopMatrix();
}

void drawBotShadow(float groundLocalY) {
    constexpr int segments = 24;
    glDisable(GL_LIGHTING);
    glColor4f(0.0f, 0.0f, 0.0f, 0.22f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0.0f, groundLocalY + 0.8f, 0.0f);
    for (int i = 0; i <= segments; ++i) {
        const float t = 2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(segments);
        glVertex3f(std::cos(t) * 46.0f, groundLocalY + 0.8f, std::sin(t) * 34.0f);
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

void drawBot(const Bot& b, float simTime) {
    const float gait = b.wobblePhase + simTime * (b.wobbleFreq * (1.0f + 0.6f * b.limpSeverity));
    float leftStride = 0.0f;
    float rightStride = 0.0f;
    float torsoBob = 0.0f;
    if (b.hasLLD) {
        leftStride = (0.50f - 0.28f * b.limpSeverity) * std::sin(gait) * 16.0f;
        rightStride = (1.00f + 0.22f * b.limpSeverity) * std::sin(gait + 2.2f) * 20.0f;
        torsoBob = std::sin(gait * 1.4f) * (2.0f + 2.5f * b.limpSeverity);
    } else {
        leftStride = 0.82f * std::sin(gait) * 15.0f;
        rightStride = 0.82f * std::sin(gait + static_cast<float>(M_PI)) * 15.0f;
        torsoBob = std::sin(gait * 1.2f) * 1.3f;
    }
    const float groundY = terrainHeight(b.x, b.y) + layerBaseHeight(b.layerPos);

    float upX = 0.0f, upY = 1.0f, upZ = 0.0f;
    terrainNormal(b.x, b.y, upX, upY, upZ);

    float fwdX = std::cos(b.heading);
    float fwdY = 0.0f;
    float fwdZ = std::sin(b.heading);
    const float fwdDotUp = fwdX * upX + fwdY * upY + fwdZ * upZ;
    fwdX -= upX * fwdDotUp;
    fwdY -= upY * fwdDotUp;
    fwdZ -= upZ * fwdDotUp;
    const float fwdLen = std::sqrt(fwdX * fwdX + fwdY * fwdY + fwdZ * fwdZ);
    if (fwdLen > 0.0001f) {
        fwdX /= fwdLen;
        fwdY /= fwdLen;
        fwdZ /= fwdLen;
    }

    float sideX = upY * fwdZ - upZ * fwdY;
    float sideY = upZ * fwdX - upX * fwdZ;
    float sideZ = upX * fwdY - upY * fwdX;
    const float sideLen = std::sqrt(sideX * sideX + sideY * sideY + sideZ * sideZ);
    if (sideLen > 0.0001f) {
        sideX /= sideLen;
        sideY /= sideLen;
        sideZ /= sideLen;
    }

    glPushMatrix();
    glTranslatef(b.x, groundY + kBotBaseY + torsoBob, b.y);
    const GLfloat orient[16] = {
        fwdX, fwdY, fwdZ, 0.0f, upX,  upY,  upZ,  0.0f,
        sideX, sideY, sideZ, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    glMultMatrixf(orient);

    const bool highDetail = cameraDistance < 8500.0f;
    const bool mediumDetail = cameraDistance < 12000.0f;

    if (highDetail) {
        drawBotShadow(-kBotBaseY - torsoBob);
    }

    if (b.hasLLD) {
        if (highDetail && botHighDisplayList != 0) {
            glCallList(botHighDisplayList);
        } else if (botLowDisplayList != 0) {
            glCallList(botLowDisplayList);
        }
    } else {
        if (highDetail && botHighOrangeDisplayList != 0) {
            glCallList(botHighOrangeDisplayList);
        } else if (botLowOrangeDisplayList != 0) {
            glCallList(botLowOrangeDisplayList);
        }
    }

    if (highDetail) {
        // Left legs (shorter/less powerful).
        if (b.hasLLD) {
            glColor3f(0.20f, 1.00f, 0.40f);
        } else {
            glColor3f(1.00f, 0.62f, 0.18f);
        }
        drawLeg(12.0f, 16.0f, leftStride, 0.72f, 2.2f);
        drawLeg(-14.0f, 13.0f, -leftStride * 0.7f, 0.66f, 2.8f);

        // Right legs (longer/stronger stride).
        if (b.hasLLD) {
            glColor3f(0.26f, 1.00f, 0.46f);
        } else {
            glColor3f(1.00f, 0.70f, 0.24f);
        }
        drawLeg(12.0f, -16.0f, rightStride, 1.0f, 0.0f);
        drawLeg(-14.0f, -13.0f, -rightStride * 0.75f, 0.96f, 0.5f);
    } else if (mediumDetail) {
        if (b.hasLLD) {
            glColor3f(0.20f, 1.00f, 0.40f);
        } else {
            glColor3f(1.00f, 0.62f, 0.18f);
        }
        drawLeg(10.0f, 14.5f, leftStride * 0.6f, 0.70f, 1.9f);
        if (b.hasLLD) {
            glColor3f(0.26f, 1.00f, 0.46f);
        } else {
            glColor3f(1.00f, 0.70f, 0.24f);
        }
        drawLeg(10.0f, -14.5f, rightStride * 0.6f, 0.95f, 0.1f);
    }

    glPopMatrix();
}

void drawTrails() {
    constexpr int stride = 1;
    constexpr size_t maxPointsPerBot = static_cast<size_t>(1) << 30;

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glLineWidth(cameraDistance > 9000.0f ? 1.0f : 1.25f);
    for (const auto& b : bots) {
        if (b.trail.size() < 2) continue;
        if (b.hasLLD) {
            glColor4f(0.24f, 1.00f, 0.44f, 0.90f);
        } else {
            glColor4f(1.00f, 0.62f, 0.20f, 0.90f);
        }
        const size_t n = b.trail.size();
        const size_t start = n > maxPointsPerBot ? n - maxPointsPerBot : 0;
        glBegin(GL_LINE_STRIP);
        for (size_t i = start; i < n; i += static_cast<size_t>(stride)) {
            const auto& p = b.trail[i];
            glVertex3f(p.x, terrainHeight(p.x, p.y) + layerBaseHeight(p.layerPos) + kTrailY, p.y);
        }
        if ((n - 1 - start) % static_cast<size_t>(stride) != 0) {
            const auto& last = b.trail.back();
            glVertex3f(last.x, terrainHeight(last.x, last.y) + layerBaseHeight(last.layerPos) + kTrailY, last.y);
        }
        glEnd();
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
}

void drawFpsOverlay() {
    char text[64];
    if (timeScale > 1.0f) {
        std::snprintf(text, sizeof(text), "FPS: %.1f  Time: x%.0f", fpsValue, timeScale);
    } else {
        std::snprintf(text, sizeof(text), "FPS: %.1f", fpsValue);
    }

    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, static_cast<double>(windowWidth), 0.0, static_cast<double>(windowHeight));

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(0.96f, 0.97f, 0.98f);
    glRasterPos2i(14, windowHeight - 24);
    for (const char* p = text; *p != '\0'; ++p) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
}

void setupCameraAndLights() {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (cameraViewMode == CameraViewMode::TopDown) {
        // Centered full-world top-down overview.
        const float targetX = 0.0f;
        const float targetZ = 0.0f;
        const float targetY = layerBaseHeight(static_cast<float>(maxCityLayers - 1) * 0.5f);
        cameraTargetX = targetX;
        cameraTargetY = targetY;
        cameraTargetZ = targetZ;
        cameraTargetCity = cityIndexForPosition(targetX, targetZ);
        cameraPosX = targetX;
        cameraPosY = targetY + cameraDistance;
        cameraPosZ = targetZ + cameraDistance * 0.01f;
        gluLookAt(targetX, targetY + cameraDistance, targetZ + cameraDistance * 0.01f, targetX, targetY, targetZ, 0.0, 0.0,
                  -1.0);
    } else if (cameraViewMode == CameraViewMode::Isometric) {
        // Center and fit the full city grid in isometric view; allow zoom via cameraDistance.
        const float targetX = 0.0f;
        const float targetZ = 0.0f;
        const float baseTargetY = layerBaseHeight(static_cast<float>(maxCityLayers - 1) * 0.5f);
        const float targetY = baseTargetY - cameraDistance * 0.16f; // stronger downward aim to center world vertically
        const double isoX = targetX + cameraDistance * 0.76;
        const double isoY = baseTargetY + cameraDistance * 0.62;
        const double isoZ = targetZ + cameraDistance * 0.76;
        cameraTargetX = targetX;
        cameraTargetY = targetY;
        cameraTargetZ = targetZ;
        cameraTargetCity = cityIndexForPosition(targetX, targetZ);
        cameraPosX = static_cast<float>(isoX);
        cameraPosY = static_cast<float>(isoY);
        cameraPosZ = static_cast<float>(isoZ);
        gluLookAt(isoX, isoY, isoZ, targetX, targetY, targetZ, 0.0, 1.0, 0.0);
    } else {
        // Side orthant view centered on the world.
        const float targetX = 0.0f;
        const float targetZ = 0.0f;
        const float targetY = layerBaseHeight(static_cast<float>(maxCityLayers - 1) * 0.5f);
        const double sideX = targetX + cameraDistance;
        const double sideY = targetY + cameraDistance * 0.08;
        const double sideZ = targetZ + cameraDistance * 0.06;
        cameraTargetX = targetX;
        cameraTargetY = targetY;
        cameraTargetZ = targetZ;
        cameraTargetCity = cityIndexForPosition(targetX, targetZ);
        cameraPosX = static_cast<float>(sideX);
        cameraPosY = static_cast<float>(sideY);
        cameraPosZ = static_cast<float>(sideZ);
        gluLookAt(sideX, sideY, sideZ, targetX, targetY, targetZ, 0.0, 1.0, 0.0);
    }

    const GLfloat light0Pos[] = {2400.0f, 8200.0f, 1800.0f, 1.0f};
    const GLfloat light0Diff[] = {1.12f, 1.10f, 1.06f, 1.0f};
    const GLfloat light1Pos[] = {-4200.0f, 3600.0f, -900.0f, 1.0f};
    const GLfloat light1Diff[] = {0.18f, 0.22f, 0.26f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light0Pos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light0Diff);
    glLightfv(GL_LIGHT1, GL_POSITION, light1Pos);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light1Diff);
}

void drawScene3D(bool withOverlay) {
    setupCameraAndLights();
    drawWorld();
    drawTrails();
    drawGoal();
    for (const auto& b : bots) {
        if (cameraViewMode == CameraViewMode::TopDown) {
            const float dx = b.x - cameraTargetX;
            const float dy = b.y - cameraTargetZ;
            if (dx * dx + dy * dy > 92000.0f * 92000.0f) continue;
        }
        drawBot(b, currentSimTime);
    }
    if (withOverlay) {
        drawFpsOverlay();
    }
}

bool exportPng10000(CameraViewMode exportMode, float exportDistance, const std::string& path) {
    constexpr int exportW = 20000;
    constexpr int exportH = 20000;
    constexpr int overlap = 4;
    const int maxTileW = std::max(128, windowWidth - 16);
    const int maxTileH = std::max(128, windowHeight - 16);
    const int tileSize = std::max(128, std::min({512, maxTileW, maxTileH}));

    std::vector<unsigned char> image(static_cast<size_t>(exportW) * static_cast<size_t>(exportH) * 3U);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    GLint prevReadBuffer = GL_BACK;
    GLint prevDrawBuffer = GL_BACK;
    glGetIntegerv(GL_READ_BUFFER, &prevReadBuffer);
    glGetIntegerv(GL_DRAW_BUFFER, &prevDrawBuffer);
    GLint prevFramebuffer = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING_EXT, &prevFramebuffer);

    GLuint fbo = 0;
    GLuint colorRb = 0;
    GLuint depthRb = 0;
    glGenFramebuffersEXT(1, &fbo);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
    glGenRenderbuffersEXT(1, &colorRb);
    glGenRenderbuffersEXT(1, &depthRb);

    bool fboOk = (fbo != 0 && colorRb != 0 && depthRb != 0);
    if (!fboOk) {
        if (depthRb != 0) glDeleteRenderbuffersEXT(1, &depthRb);
        if (colorRb != 0) glDeleteRenderbuffersEXT(1, &colorRb);
        if (fbo != 0) glDeleteFramebuffersEXT(1, &fbo);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, prevFramebuffer);
        glReadBuffer(prevReadBuffer);
        glDrawBuffer(prevDrawBuffer);
        return false;
    }

    const CameraViewMode prevMode = cameraViewMode;
    const float prevDistance = cameraDistance;
    cameraViewMode = exportMode;
    cameraDistance = exportDistance;
    exportRenderPass = true;

    int allocW = 0;
    int allocH = 0;
    bool ok = true;

    for (int yb = 0; yb < exportH; yb += tileSize) {
        if (!ok) break;
        const int outH = std::min(tileSize, exportH - yb);
        for (int x = 0; x < exportW; x += tileSize) {
            if (!ok) break;
            const int outW = std::min(tileSize, exportW - x);

            const int padL = (x == 0) ? 0 : overlap;
            const int padR = (x + outW >= exportW) ? 0 : overlap;
            const int padB = (yb == 0) ? 0 : overlap;
            const int padT = (yb + outH >= exportH) ? 0 : overlap;

            const int rx = x - padL;
            const int ry = yb - padB;
            const int rw = outW + padL + padR;
            const int rh = outH + padB + padT;

            std::vector<unsigned char> tileBuf(static_cast<size_t>(rw) * static_cast<size_t>(rh) * 3U);

            if (rw != allocW || rh != allocH) {
                allocW = rw;
                allocH = rh;
                glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, colorRb);
                glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGBA8, allocW, allocH);
                glFramebufferRenderbufferEXT(
                    GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, colorRb);

                glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthRb);
                glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, allocW, allocH);
                glFramebufferRenderbufferEXT(
                    GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthRb);

                const GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
                if (status != GL_FRAMEBUFFER_COMPLETE_EXT) {
                    ok = false;
                    break;
                }
            }

            glViewport(0, 0, rw, rh);
            setProjectionTiled(exportW, exportH, rx, ry, rw, rh);
            glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
            glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            drawScene3D(false);
            glFinish();
            glReadPixels(0, 0, rw, rh, GL_RGB, GL_UNSIGNED_BYTE, tileBuf.data());

            for (int row = 0; row < outH; ++row) {
                const int dstY = exportH - 1 - (yb + row);
                const size_t dst =
                    (static_cast<size_t>(dstY) * static_cast<size_t>(exportW) + static_cast<size_t>(x)) * 3U;
                const size_t src = (static_cast<size_t>(row + padB) * static_cast<size_t>(rw) +
                                    static_cast<size_t>(padL)) *
                                   3U;
                std::memcpy(&image[dst], &tileBuf[src], static_cast<size_t>(outW) * 3U);
            }
        }
    }

    if (ok) {
        ok = writePngRgb(path, exportW, exportH, image);
    }
    exportRenderPass = false;
    cameraViewMode = prevMode;
    cameraDistance = prevDistance;
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, prevFramebuffer);
    glReadBuffer(prevReadBuffer);
    glDrawBuffer(prevDrawBuffer);
    glDeleteRenderbuffersEXT(1, &depthRb);
    glDeleteRenderbuffersEXT(1, &colorRb);
    glDeleteFramebuffersEXT(1, &fbo);

    glViewport(0, 0, windowWidth, windowHeight);
    setProjection(windowWidth, windowHeight);
    glutPostRedisplay();
    return ok;
}

bool exportPng10000Isometric() {
    return exportPng10000(CameraViewMode::Isometric, isometricFitDistance(), "export_20000x20000.png");
}

bool exportPng10000TopDown() {
    const float topDownFit = std::max(kMinCameraDistance, std::min(kMaxCameraDistance, isometricFitDistance() * 0.98f));
    return exportPng10000(CameraViewMode::TopDown, topDownFit, "export_topdown_20000x20000.png");
}

bool exportPng10000Side() {
    const float sideFit =
        std::max(kMinCameraDistance, std::min(kMaxCameraDistance, sideFitDistance() * 1.12f));
    return exportPng10000(CameraViewMode::Side, sideFit, "export_side_20000x20000.png");
}

void updateBots(float dt, float simTime) {
    const float nominalSpeed = kStandardSpeed * kBotSpeedFactor;

    for (auto& b : bots) {
        const float prevX = b.x;
        const float prevY = b.y;

        b.stateTimer += dt;
        if (!b.onSkyway) {
            b.city = cityIndexForPosition(b.x, b.y);
        }

        // Handle active skyway transit first.
        if (b.onSkyway) {
            b.skywayProgress += dt / 6.0f;
            if (b.skywayProgress >= 1.0f) {
                b.skywayProgress = 1.0f;
                b.onSkyway = false;
                b.city = b.skywayToCity;
                b.x = b.skywayEndX;
                b.y = b.skywayEndY;
            } else {
                const float t = b.skywayProgress;
                b.x = b.skywayStartX + (b.skywayEndX - b.skywayStartX) * t;
                b.y = b.skywayStartY + (b.skywayEndY - b.skywayStartY) * t;
            }
            b.layerPos = static_cast<float>(b.layer);
        }

        float rampCX = cityCenters[b.city][0];
        float rampCY = cityCenters[b.city][1];

        const int targetCity = goalCity;
        const int targetLayer = goalLayer;
        int layerDiff = targetLayer - b.layer;
        float targetX = goalX;
        float targetY = goalY;
        const bool cityDiff = (b.city != targetCity);

        if (cityDiff) {
            const int nextCity = nextCityOnPath(b.city, targetCity);
            if (nextCity == b.city) {
                // Fallback safety if graph/pathing fails: just head to goal city center directly.
                targetX = cityCenters[targetCity][0];
                targetY = cityCenters[targetCity][1];
            }
            float sx = 0.0f, sy = 0.0f, ex = 0.0f, ey = 0.0f;
            skywayEndpoints(b.city, nextCity, sx, sy, ex, ey);
            const int transferLayer = std::min({b.layer, cityLayerCounts[b.city] - 1, cityLayerCounts[nextCity] - 1});

            if (!b.onSkyway && b.layer == transferLayer) {
                targetX = sx;
                targetY = sy;
                const float dxS = b.x - sx;
                const float dyS = b.y - sy;
                if (dxS * dxS + dyS * dyS < 180.0f * 180.0f) {
                    b.onSkyway = true;
                    b.skywayFromCity = b.city;
                    b.skywayToCity = nextCity;
                    b.skywayProgress = 0.0f;
                    b.skywayStartX = sx;
                    b.skywayStartY = sy;
                    b.skywayEndX = ex;
                    b.skywayEndY = ey;
                    b.stateTimer = 0.0f;
                }
            } else if (b.layer != transferLayer) {
                targetX = rampCX;
                targetY = rampCY;
                layerDiff = transferLayer - b.layer;
            }
        }

        if (!cityDiff && !b.onSkyway && layerDiff != 0) {
            // Route to city ramp if goal is on a different layer.
            targetX = rampCX;
            targetY = rampCY;
        }

        if (!b.onSkyway && !b.onRamp && layerDiff != 0) {
            const float dxC = b.x - rampCX;
            const float dyC = b.y - rampCY;
            const float rToCenter = std::sqrt(dxC * dxC + dyC * dyC);
            if (rToCenter >= kRampInnerRadius && rToCenter <= kRampOuterRadius && unit01(rng) < dt * 1.8f) {
                b.onRamp = true;
                b.rampFromLayer = b.layer;
                b.rampToLayer = b.layer + ((layerDiff > 0) ? 1 : -1);
                if (b.rampToLayer < 0) b.rampToLayer = 0;
                if (b.rampToLayer >= cityLayerCounts[b.city]) b.rampToLayer = cityLayerCounts[b.city] - 1;
                b.rampProgress = 0.0f;
                b.stateTimer = 0.0f;
            }
        }

        if (b.onRamp && !b.onSkyway) {
            float r = std::sqrt((b.x - rampCX) * (b.x - rampCX) + (b.y - rampCY) * (b.y - rampCY));
            if (r < 1.0f) r = 1.0f;
            const float nx = (b.x - rampCX) / r;
            const float ny = (b.y - rampCY) / r;
            const float tx = -ny;
            const float ty = nx;
            const float rampDir = (b.rampToLayer > b.rampFromLayer) ? 1.0f : -1.0f;

            // Pull onto ramp centerline and move tangentially.
            const float targetRX = rampCX + nx * kRampCenterRadius;
            const float targetRY = rampCY + ny * kRampCenterRadius;
            b.x += (targetRX - b.x) * std::min(1.0f, dt * 2.8f);
            b.y += (targetRY - b.y) * std::min(1.0f, dt * 2.8f);

            const float rampSpeed = (2.0f * static_cast<float>(M_PI) * kRampCenterRadius) / kRampTravelSeconds;
            b.x += tx * rampDir * rampSpeed * dt;
            b.y += ty * rampDir * rampSpeed * dt;
            b.heading = wrapAngle(std::atan2(ty * rampDir, tx * rampDir));

            b.rampProgress += dt / kRampTravelSeconds;
            if (b.rampProgress >= 1.0f) {
                b.rampProgress = 1.0f;
                b.layer = b.rampToLayer;
                b.layerPos = static_cast<float>(b.layer);
                b.onRamp = false;
                b.stateTimer = 0.0f;
            } else {
                b.layerPos = static_cast<float>(b.rampFromLayer) +
                             (static_cast<float>(b.rampToLayer - b.rampFromLayer) * b.rampProgress);
            }
        } else {
            b.layerPos = static_cast<float>(b.layer);
        }

        const float toGoalX = targetX - b.x;
        const float toGoalY = targetY - b.y;
        const float desired = std::atan2(toGoalY, toGoalX);
        const float delta = wrapAngle(desired - b.heading);

        // Easier left turns than right turns.
        const float maxTurn = (delta >= 0.0f ? kTurnRateLeft : kTurnRateRight) * dt;
        const float appliedTurn = clampAbs(delta, maxTurn);

        // Gait asymmetry from leg length discrepancy.
        const float gait = b.wobblePhase + simTime * b.wobbleFreq;
        const float leftGait = std::sin(gait);
        const float rightGait = std::sin(gait + static_cast<float>(M_PI) * (0.65f + 0.2f * b.limpSeverity));

        float wobble = 0.0f;
        float controlNoise = 0.0f;
        float gaitYaw = 0.0f;
        float passiveLeftDrift = 0.0f;
        float constantLeftPull = 0.0f;
        const float dirX = std::cos(b.heading);
        const float dirY = std::sin(b.heading);
        const float leftX = -dirY;
        const float leftY = dirX;
        const float sampleDist = 110.0f;
        const float sideDist = 85.0f;
        const float roughForward = terrainRoughness(b.x + dirX * sampleDist, b.y + dirY * sampleDist);
        const float roughLeft = terrainRoughness(b.x + leftX * sideDist, b.y + leftY * sideDist);
        const float roughRight = terrainRoughness(b.x - leftX * sideDist, b.y - leftY * sideDist);
        const float roughHere = terrainRoughness(b.x, b.y);
        const float flatSteer = (roughRight - roughLeft) * (1.7f + 1.2f * b.limpSeverity) +
                                (roughForward - roughHere) * 0.65f;
        float meander = 0.0f;
        float flutter = 0.0f;
        float organicInput = 0.0f;

        if (b.hasLLD) {
            // Wobble and asymmetry from leg-length discrepancy.
            wobble = (kWobbleAmplitude + 0.2f * b.limpSeverity) * leftGait;
            controlNoise = (0.09f + 0.06f * b.limpSeverity) * noiseDist(rng);
            gaitYaw = (rightGait * b.rightLegPower - leftGait * b.leftLegPower) * (0.22f + 0.18f * b.limpSeverity);
            passiveLeftDrift = b.leftTurnBias * (0.4f + 0.6f * (0.5f + 0.5f * leftGait));
            constantLeftPull = kAlwaysLeftPull * (0.95f + 1.05f * b.limpSeverity);
            meander = std::sin(b.wanderPhase + simTime * b.wanderFreq) * b.wanderStrength * (0.9f + 0.7f * b.limpSeverity);
            flutter = 0.14f * std::sin((b.wanderPhase * 1.7f) + simTime * (b.wanderFreq * 4.2f + 0.9f));
            organicInput = meander + flutter + 0.06f * noiseDist(rng);
        } else {
            // Orange bots: no LLD (no left pull/asymmetric gait), smoother control.
            wobble = 0.06f * std::sin(gait * 0.9f);
            controlNoise = 0.02f * noiseDist(rng);
            meander = std::sin(b.wanderPhase + simTime * b.wanderFreq) * b.wanderStrength;
            flutter = 0.03f * std::sin((b.wanderPhase * 1.4f) + simTime * (b.wanderFreq * 2.6f + 0.6f));
            organicInput = meander + flutter + 0.01f * noiseDist(rng);
        }
        b.turnMemory = 0.90f * b.turnMemory + 0.10f * organicInput;
        const float organicSteer = b.hasLLD ? (b.turnMemory * (1.0f + 0.6f * b.limpSeverity)) : (b.turnMemory * 0.45f);

        if (!b.onRamp && !b.onSkyway) {
            b.heading = wrapAngle(
                b.heading + appliedTurn +
                (wobble + controlNoise + gaitYaw + passiveLeftDrift + constantLeftPull + flatSteer + organicSteer) * dt);
        }

        const float wobblePenalty = 1.0f - kWobbleVelocityPenalty * std::fabs(wobble);
        float strideDrive = 1.0f;
        float stumblePenalty = 1.0f;
        float roughnessPenalty = 1.0f;
        float speedJitter = 1.0f;
        float gaitLurch = 1.0f;
        float speed = 0.0f;

        if (b.hasLLD) {
            strideDrive =
                0.45f + 0.65f * std::max(0.0f, rightGait * b.rightLegPower) + 0.28f * std::max(0.0f, leftGait * b.leftLegPower);
            stumblePenalty = 1.0f - 0.32f * b.limpSeverity * (0.5f + 0.5f * std::fabs(leftGait - rightGait));
            roughnessPenalty = 1.0f - std::min(0.45f, roughHere * (0.70f + 0.55f * b.limpSeverity));
            speedJitter = 0.70f + 0.30f * unit01(rng);
            gaitLurch = 0.84f + 0.28f * std::max(0.0f, std::sin(simTime * (b.wanderFreq * 2.1f) + b.wanderPhase));
            speed =
                nominalSpeed * b.speedMultiplier * kEfficiencyLoss * wobblePenalty * strideDrive * stumblePenalty * roughnessPenalty *
                speedJitter * gaitLurch;
        } else {
            strideDrive = 0.96f + 0.08f * std::sin(gait + b.wanderPhase * 0.5f);
            roughnessPenalty = 1.0f - std::min(0.25f, roughHere * 0.35f);
            speedJitter = 0.96f + 0.08f * unit01(rng);
            speed = nominalSpeed * b.speedMultiplier * wobblePenalty * strideDrive * roughnessPenalty * speedJitter;
        }

        if (!b.onRamp && !b.onSkyway) {
            b.x += std::cos(b.heading) * speed * dt;
            b.y += std::sin(b.heading) * speed * dt;
        }

        // Transit watchdogs: recover from pathological stuck states.
        if (b.onSkyway && b.stateTimer > 10.0f) {
            b.onSkyway = false;
            b.city = cityIndexForPosition(b.x, b.y);
            b.stateTimer = 0.0f;
        }
        if (b.onRamp && b.stateTimer > 12.0f) {
            b.onRamp = false;
            b.layerPos = static_cast<float>(b.layer);
            b.stateTimer = 0.0f;
        }

        // Keep bots inside circular world.
        const float d = std::sqrt(b.x * b.x + b.y * b.y);
        if (d > kWorldRadius - kBotBodyLength) {
            const float nx = b.x / d;
            const float ny = b.y / d;
            b.x = nx * (kWorldRadius - kBotBodyLength);
            b.y = ny * (kWorldRadius - kBotBodyLength);

            // Bounce inward with slight left-turn bias.
            const float inward = std::atan2(-ny, -nx);
            b.heading = wrapAngle(inward + 0.2f);
        }

        // Global anti-stall: if almost no displacement for several seconds, nudge and reset transit.
        const float moved2 = (b.x - prevX) * (b.x - prevX) + (b.y - prevY) * (b.y - prevY);
        if (moved2 < 0.25f) {
            b.stallTimer += dt;
        } else {
            b.stallTimer = 0.0f;
        }
        if (b.stallTimer > 3.0f) {
            b.onSkyway = false;
            b.onRamp = false;
            b.heading = wrapAngle(b.heading + 0.8f + 0.6f * noiseDist(rng));
            b.x += (25.0f + 40.0f * unit01(rng)) * std::cos(b.heading);
            b.y += (25.0f + 40.0f * unit01(rng)) * std::sin(b.heading);
            b.city = cityIndexForPosition(b.x, b.y);
            b.layer = std::max(0, std::min(cityLayerCounts[b.city] - 1, b.layer));
            b.layerPos = static_cast<float>(b.layer);
            b.stateTimer = 0.0f;
            b.stallTimer = 0.0f;
        }

        // Keep a permanent trail with modest spatial sampling.
        if (b.trail.empty()) {
            b.trail.push_back({b.x, b.y, b.layerPos});
        } else {
            const float lastX = b.trail.back().x;
            const float lastY = b.trail.back().y;
            const float dx = b.x - lastX;
            const float dy = b.y - lastY;
            if (dx * dx + dy * dy >= kTrailMinPointDistance * kTrailMinPointDistance) {
                b.trail.push_back({b.x, b.y, b.layerPos});
            }
        }
    }
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    setProjection(windowWidth, windowHeight);
    drawScene3D(true);

    glutSwapBuffers();
}

void reshape(int w, int h) {
    windowWidth = w;
    windowHeight = h;
    glViewport(0, 0, w, h);
    setProjection(w, h);
    glLoadIdentity();
}

void idle() {
    const float now = 0.001f * static_cast<float>(glutGet(GLUT_ELAPSED_TIME));
    if (now < nextFrameTime) return;
    nextFrameTime = now + kFrameInterval;
    currentSimTime = now;
    float dt = now - previousTime;
    previousTime = now;

    if (dt < 0.0f) dt = 0.0f;
    if (dt > 0.05f) dt = 0.05f;
    dt *= timeScale;

    if (now - lastGoalSpawnTime >= kGoalRespawnSeconds) {
        spawnGoal();
        lastGoalSpawnTime = now;
    }

    updateBots(dt, now);

    fpsFrameCount += 1;
    fpsAccumTime += dt;
    if (fpsAccumTime >= 0.5f) {
        fpsValue = static_cast<double>(fpsFrameCount) / static_cast<double>(fpsAccumTime);
        fpsFrameCount = 0;
        fpsAccumTime = 0.0f;
    }

    glutPostRedisplay();
}

void mouseButton(int button, int state, int x, int y) {
    (void)x;
    if (button == 3 && state == GLUT_DOWN) {
        applyZoom(-650.0f); // wheel up
    } else if (button == 4 && state == GLUT_DOWN) {
        applyZoom(650.0f); // wheel down
    } else if (button == GLUT_RIGHT_BUTTON || button == GLUT_MIDDLE_BUTTON) {
        if (state == GLUT_DOWN) {
            zoomDragActive = true;
            zoomDragButton = button;
            lastMouseY = y;
        } else {
            if (zoomDragButton == button) {
                zoomDragActive = false;
                zoomDragButton = -1;
            }
        }
    }
}

void mouseMotion(int x, int y) {
    (void)x;
    if (!zoomDragActive) return;
    const int dy = y - lastMouseY;
    lastMouseY = y;
    applyZoom(static_cast<float>(dy) * 35.0f);
}

void keyboard(unsigned char key, int x, int y) {
    (void)x;
    (void)y;
    if (key == '1') {
        cameraViewMode = CameraViewMode::TopDown;
        cameraDistance = std::max(kMinCameraDistance, std::min(kMaxCameraDistance, isometricFitDistance() * 0.98f));
        glutPostRedisplay();
    } else if (key == '2') {
        cameraViewMode = CameraViewMode::Isometric;
        cameraDistance = std::max(kMinCameraDistance, std::min(kMaxCameraDistance, isometricFitDistance() * 0.93f));
        glutPostRedisplay();
    } else if (key == '3') {
        cameraViewMode = CameraViewMode::Side;
        cameraDistance = std::max(kMinCameraDistance, std::min(kMaxCameraDistance, sideFitDistance()));
        glutPostRedisplay();
    } else if (key == 'e' || key == 'E') {
        std::printf("Exporting 20000x20000 PNG...\n");
        const bool ok = exportPng10000Isometric();
        std::printf("%s\n", ok ? "Export complete: export_20000x20000.png" : "Export failed");
    } else if (key == 'w' || key == 'W') {
        std::printf("Exporting top-down 20000x20000 PNG...\n");
        const bool ok = exportPng10000TopDown();
        std::printf("%s\n", ok ? "Export complete: export_topdown_20000x20000.png" : "Export failed");
    } else if (key == 'r' || key == 'R') {
        std::printf("Exporting side-view 20000x20000 PNG...\n");
        const bool ok = exportPng10000Side();
        std::printf("%s\n", ok ? "Export complete: export_side_20000x20000.png" : "Export failed");
    } else if (key == 't' || key == 'T') {
        timeScale = (timeScale > 1.0f) ? 1.0f : 10.0f;
        glutPostRedisplay();
    }
}

void init() {
    glClearColor(0.015f, 0.018f, 0.022f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_SMOOTH);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    const GLfloat ambient[] = {0.10f, 0.10f, 0.12f, 1.0f};
    const GLfloat specular[] = {0.42f, 0.44f, 0.46f, 1.0f};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 34.0f);
    rng.seed(static_cast<unsigned int>(std::time(nullptr)));
    initCityLayout();
    initCityLayers();
    createGroundTexture();
    createDisplayLists();
    initBots();
    spawnGoal();
    cameraViewMode = CameraViewMode::TopDown;
    cameraDistance = std::max(kMinCameraDistance, std::min(kMaxCameraDistance, isometricFitDistance() * 0.98f));

    previousTime = 0.001f * static_cast<float>(glutGet(GLUT_ELAPSED_TIME));
    lastGoalSpawnTime = previousTime;
    nextFrameTime = previousTime;
}
} // namespace

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(1280, 900);
    glutCreateWindow("Circular World Bots (3D OpenGL)");

    init();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMotion);
    glutKeyboardFunc(keyboard);

    glutMainLoop();
    return 0;
}
