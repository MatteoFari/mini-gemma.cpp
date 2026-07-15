#pragma once
#include <random>
#include <span>
int sample(std::span<const float> logits, float temperature, float top_p, std::mt19937 &rng);
