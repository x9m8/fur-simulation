import * as THREE from 'three';
import { GPUComputationRenderer } from 'three/examples/jsm/misc/GPUComputationRenderer.js';

const SIM_SIZE = 128; // 128x128 = 16,384 parallel GPU spring-verlet hair patches & wind simulation cells

/**
 * GPGPU Wind Simulation Shader
 * Simulates a dynamic 2D spherical aerodynamic wind velocity field with:
 * 1. Macro directional wind flow deflected tangentially around the creature's curved body (potential flow)
 * 2. 3D divergence-free curl-noise turbulence creating organic vortex shedding on the wake/leeward sides
 * 3. Traveling ripple wave propagation (Kelvin-Helmholtz surface instability) producing moving fur wave-fronts
 * 4. Procedural gust envelope pulses with advection momentum transport
 * Output: vec4(windVelocity.xyz, rippleAmplitude)
 */
const WIND_FRAGMENT_SHADER = /* glsl */ `
  uniform float uTime;
  uniform float uDelta;
  uniform float uWindStrength;
  uniform vec3 uWindDirection;
  uniform float uWindGustiness;
  uniform float uWindFrequency;
  uniform float uWindSpeed;
  uniform vec3 uCreatureVelocity;

  // Compact 3D potential field for divergence-free curl turbulence
  vec3 potentialField(vec3 p, float t) {
    float f1 = sin(p.y * 2.3 + t * 1.8) * cos(p.z * 1.9 + t * 1.4) + 0.5 * sin(p.z * 4.7 - t * 2.6);
    float f2 = cos(p.x * 2.5 + t * 1.6) * sin(p.z * 2.1 + t * 1.2) + 0.5 * cos(p.x * 4.9 - t * 2.8);
    float f3 = sin(p.x * 2.1 - t * 1.7) * cos(p.y * 2.6 + t * 1.5) + 0.5 * sin(p.y * 4.3 + t * 2.5);
    return vec3(f1, f2, f3);
  }

  // Exact finite-difference curl: ensures div(v) = 0 so wind flows smoothly without compression spikes
  vec3 curlNoise(vec3 p, float t) {
    float eps = 0.09;
    vec3 dx = vec3(eps, 0.0, 0.0);
    vec3 dy = vec3(0.0, eps, 0.0);
    vec3 dz = vec3(0.0, 0.0, eps);

    vec3 px1 = potentialField(p + dx, t);
    vec3 px0 = potentialField(p - dx, t);
    vec3 py1 = potentialField(p + dy, t);
    vec3 py0 = potentialField(p - dy, t);
    vec3 pz1 = potentialField(p + dz, t);
    vec3 pz0 = potentialField(p - dz, t);

    float dFz_dy = (py1.z - py0.z) / (2.0 * eps);
    float dFy_dz = (pz1.y - pz0.y) / (2.0 * eps);

    float dFx_dz = (pz1.x - pz0.x) / (2.0 * eps);
    float dFz_dx = (px1.z - px0.z) / (2.0 * eps);

    float dFy_dx = (px1.y - px0.y) / (2.0 * eps);
    float dFx_dy = (py1.x - py0.x) / (2.0 * eps);

    return vec3(dFz_dy - dFy_dz, dFx_dz - dFz_dx, dFy_dx - dFx_dy);
  }

  void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;

    // Reconstruct 3D surface normal from spherical simulation texel
    float theta = (uv.x * 2.0 - 1.0) * 3.14159265;
    float phi = (1.0 - uv.y) * 3.14159265;
    vec3 surfNormal = normalize(vec3(
      sin(phi) * sin(theta),
      cos(phi),
      sin(phi) * cos(theta)
    ));

    // Sample previous frame's wind simulation state for advection and momentum continuity
    vec4 prevWind = texture2D(textureWind, uv);
    vec3 prevVel = prevWind.xyz;
    float prevRipple = prevWind.w;

    float dt = clamp(uDelta, 0.001, 0.033);

    // 1. Apparent wind: combines global wind with creature's own translational movement
    vec3 baseWindDir = normalize(uWindDirection + vec3(1e-4, 0.0, 0.0));
    vec3 apparentWind = baseWindDir * uWindStrength - uCreatureVelocity * 0.75;
    float apparentSpeed = length(apparentWind);
    vec3 effWindDir = apparentSpeed > 1e-4 ? apparentWind / apparentSpeed : baseWindDir;

    // 2. Aerodynamic flow around obstacle: project wind tangentially onto creature surface
    // Bernoulli acceleration: air speeds up along lateral flanks where dot(surfNormal, effWindDir) ~ 0
    float normalDot = dot(surfNormal, effWindDir);
    float flankSpeedBoost = 1.0 + 0.45 * (1.0 - abs(normalDot));
    vec3 tangentFlow = effWindDir - surfNormal * normalDot;
    tangentFlow = normalize(tangentFlow + vec3(1e-5, 0.0, 0.0)) * (apparentSpeed * flankSpeedBoost);

    // 3. Multi-harmonic traveling ripple wave equation (Kelvin-Helmholtz surface wave fronts)
    // Traveling coordinate along the wind direction across the spherical surface
    float travelCoord = dot(surfNormal, effWindDir);
    float wavePhase = travelCoord * (uWindFrequency * 4.2) - uTime * (uWindSpeed * 3.8);

    // Primary rolling swell wave
    float w1 = sin(wavePhase);
    // Secondary traveling ripple harmonic with phase-skewed micro-ripples
    float w2 = 0.52 * sin(wavePhase * 2.18 + cos(surfNormal.y * 5.2 + uTime * 2.4));
    // High-frequency capillary flutter ripple
    float w3 = 0.28 * sin(wavePhase * 4.75 - uTime * 6.2);

    float combinedRipple = (w1 + w2 + w3) * (0.65 + 0.35 * uWindGustiness);

    // 4. Procedural Gust Wave Pulse Envelope (sweeps across the body in periodic surges)
    float gustCycle = uTime * 0.65;
    float gustPulse = smoothstep(0.15, 0.85, 0.5 + 0.5 * sin(gustCycle) * cos(gustCycle * 0.42 + 1.2));
    float gustSurge = 1.0 + uWindGustiness * (gustPulse * 1.65 - 0.45);

    // 5. 3D Divergence-Free Curl Eddies on the Leeward / Wake Side (flow separation behind the creature)
    float wakeMask = smoothstep(-0.25, 0.75, normalDot); // leeward side
    vec3 vortices = curlNoise(surfNormal * 2.2 + vec3(0.0, uTime * 0.3, 0.0), uTime * (0.8 + 0.6 * uWindSpeed));
    // Ensure turbulence stays on the tangent plane
    vec3 tangVortices = vortices - surfNormal * dot(vortices, surfNormal);

    // 6. Upwind back-advection: sample upstream momentum
    vec2 advUV = uv - vec2(tangentFlow.x, tangentFlow.z) * (dt * 0.035 * uWindSpeed);
    vec4 advectedWind = texture2D(textureWind, fract(advUV));

    // Combine aerodynamic tangential flow, gust surge, curl turbulence, and ripple displacements
    vec3 targetWindVel = tangentFlow * gustSurge + tangVortices * (uWindStrength * uWindGustiness * (0.4 + 0.6 * wakeMask));
    
    // Add wave ripple impulse along the tangent direction
    targetWindVel += tangentFlow * (combinedRipple * 0.32 * (0.4 + 0.6 * gustSurge));

    // Smooth relaxation toward target wind velocity with advective inertia
    vec3 nextWindVel = mix(advectedWind.xyz, targetWindVel, clamp(dt * 12.0, 0.05, 0.95));
    float nextRipple = mix(prevRipple, combinedRipple * gustSurge, clamp(dt * 14.0, 0.05, 0.95));

    gl_FragColor = vec4(nextWindVel, nextRipple);
  }
`;

const VELOCITY_FRAGMENT_SHADER = /* glsl */ `
  uniform float uTime;
  uniform float uDelta;
  uniform float uStiffness;
  uniform float uDamping;
  uniform float uGravity;
  uniform float uWindStrength;
  uniform vec3 uCreatureVelocity;
  uniform vec3 uCreatureAngularVel;
  uniform vec3 uPointerLocalPos;
  uniform vec3 uPointerVelocity;
  uniform float uPointerActive;
  uniform float uBoopShockwave;

  void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;

    // Reconstruct spherical surface direction for this simulation texel
    float theta = (uv.x * 2.0 - 1.0) * 3.14159265;
    float phi = (1.0 - uv.y) * 3.14159265;
    vec3 surfNormal = normalize(vec3(
      sin(phi) * sin(theta),
      cos(phi),
      sin(phi) * cos(theta)
    ));

    vec4 posData = texture2D(texturePosition, uv);
    vec4 velData = texture2D(textureVelocity, uv);
    vec4 windData = texture2D(textureWind, uv);

    vec3 disp = posData.xyz;
    float combMemory = posData.w;
    vec3 vel = velData.xyz;

    vec3 gpgpuWind = windData.xyz;
    float windRipple = windData.w;

    float dt = clamp(uDelta, 0.001, 0.033);

    // 1. Hookean angular spring restoring force toward rest pose
    vec3 force = -uStiffness * disp;

    // 2. Tangential gravity droop (prevents hair from collapsing inward into the body)
    vec3 gravVec = vec3(0.0, -uGravity, 0.0);
    vec3 tangentGravity = gravVec - surfNormal * min(0.0, dot(gravVec, surfNormal));
    force += tangentGravity;

    // 3. Creature movement & rotational inertia (secondary motion follow-through)
    vec3 linearInertia = -uCreatureVelocity * 1.85;
    vec3 rotationalInertia = -cross(uCreatureAngularVel, surfNormal) * 2.2;
    force += linearInertia + rotationalInertia;

    // 4. GPGPU Wind Simulation Forces:
    // a. Aerodynamic Drag Force: F_drag = 0.5 * Cd * (v_wind - v_hair) * |v_wind - v_hair|
    vec3 relVel = gpgpuWind * 1.25 - vel;
    float relSpeed = length(relVel);
    vec3 dragForce = relVel * (0.85 + 0.35 * relSpeed);

    // b. Aerodynamic Flutter & Traveling Ripple Impulse:
    // Pushes hair tips along the instantaneous rippling wave front with lateral flutter lift
    vec3 flutterAxis = normalize(cross(surfNormal, gpgpuWind + vec3(1e-4, 0.0, 0.0)));
    vec3 rippleImpulse = (gpgpuWind * 0.65 + flutterAxis * (0.35 * length(gpgpuWind))) * (windRipple * 3.2);

    force += dragForce + rippleImpulse;

    // 5. Interactive pointer petting, parting, and combing
    if (uPointerActive > 0.5) {
      float dist = distance(surfNormal, normalize(uPointerLocalPos + vec3(0.0, 0.0, 1e-5)));
      if (dist < 0.55) {
        float falloff = pow(1.0 - dist / 0.55, 2.0);
        vec3 partDir = normalize(surfNormal - normalize(uPointerLocalPos) + vec3(0.0, 1e-4, 0.0));
        // Project parting & cursor velocity onto tangent plane
        vec3 strokeImpulse = uPointerVelocity * 2.8 + partDir * 3.4;
        strokeImpulse -= surfNormal * dot(strokeImpulse, surfNormal);
        force += strokeImpulse * falloff * 18.0;
      }
    }

    // 6. Click-to-boop radial shockwave ripple
    if (uBoopShockwave > 0.01) {
      float waveFront = (1.0 - uBoopShockwave) * 2.2;
      float dFront = abs(distance(surfNormal, vec3(0.0, 0.1, 0.98)) - waveFront);
      if (dFront < 0.35) {
        float ripple = (1.0 - dFront / 0.35) * uBoopShockwave;
        vec3 radial = normalize(surfNormal - vec3(0.0, 0.1, 0.98) + vec3(0.0, 1e-4, 0.0));
        force += radial * ripple * 22.0;
      }
    }

    // Semi-implicit Euler / Verlet velocity integration with exponential damping
    vel = (vel + force * dt) * exp(-uDamping * dt);

    // Clamp max velocity for guaranteed numerical stability
    float speed = length(vel);
    if (speed > 6.0) {
      vel = (vel / speed) * 6.0;
    }

    gl_FragColor = vec4(vel, combMemory);
  }
`;

const POSITION_FRAGMENT_SHADER = /* glsl */ `
  uniform float uDelta;
  uniform vec3 uPointerLocalPos;
  uniform float uPointerActive;

  void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    float theta = (uv.x * 2.0 - 1.0) * 3.14159265;
    float phi = (1.0 - uv.y) * 3.14159265;
    vec3 surfNormal = normalize(vec3(
      sin(phi) * sin(theta),
      cos(phi),
      sin(phi) * cos(theta)
    ));

    vec4 posData = texture2D(texturePosition, uv);
    vec4 velData = texture2D(textureVelocity, uv);
    vec3 disp = posData.xyz;
    float combMemory = posData.w;
    vec3 vel = velData.xyz;

    float dt = clamp(uDelta, 0.001, 0.033);
    disp += vel * dt;

    // Prevent hair from penetrating inward along the surface normal
    float inward = dot(disp, surfNormal);
    if (inward < -0.15) {
      disp -= surfNormal * (inward + 0.15);
    }

    // Elastic length constraint (prevents infinite stretching under extreme forces)
    float maxDisp = 0.85;
    float dLen = length(disp);
    if (dLen > maxDisp) {
      disp = (disp / dLen) * maxDisp;
    }

    // Track recently petted/groomed regions in alpha channel for subtle sheen boost
    combMemory *= exp(-dt * 1.2);
    if (uPointerActive > 0.5) {
      float dist = distance(surfNormal, normalize(uPointerLocalPos + vec3(0.0, 0.0, 1e-5)));
      if (dist < 0.45) {
        combMemory = min(1.0, combMemory + (1.0 - dist / 0.45) * dt * 8.0);
      }
    }

    gl_FragColor = vec4(disp, combMemory);
  }
`;

export function createGPUFurPhysics(renderer) {
  let gpuCompute = null;
  let windVariable = null;
  let posVariable = null;
  let velVariable = null;
  let fallbackTexture = null;
  let fallbackWindTexture = null;
  let isGPGPUActive = false;

  // Safe fallback textures in case WebGL context lacks float RTT
  const fallbackData = new Float32Array(4 * 4 * 4);
  fallbackTexture = new THREE.DataTexture(
    fallbackData,
    4,
    4,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  fallbackTexture.needsUpdate = true;

  const fallbackWindData = new Float32Array(4 * 4 * 4);
  fallbackWindTexture = new THREE.DataTexture(
    fallbackWindData,
    4,
    4,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  fallbackWindTexture.needsUpdate = true;

  try {
    gpuCompute = new GPUComputationRenderer(SIM_SIZE, SIM_SIZE, renderer);
    if (!renderer.capabilities.isWebGL2) {
      gpuCompute.setDataType(THREE.HalfFloatType);
    } else {
      gpuCompute.setDataType(THREE.HalfFloatType);
    }

    const dtWind = gpuCompute.createTexture();
    const dtPosition = gpuCompute.createTexture();
    const dtVelocity = gpuCompute.createTexture();

    // 1. Wind variable: GPGPU wind simulation
    windVariable = gpuCompute.addVariable('textureWind', WIND_FRAGMENT_SHADER, dtWind);
    // 2. Velocity variable: hair springs driven by GPGPU wind + inertia + gravity
    velVariable = gpuCompute.addVariable('textureVelocity', VELOCITY_FRAGMENT_SHADER, dtVelocity);
    // 3. Position variable: hair strand verlet integration
    posVariable = gpuCompute.addVariable('texturePosition', POSITION_FRAGMENT_SHADER, dtPosition);

    gpuCompute.setVariableDependencies(windVariable, [windVariable]);
    gpuCompute.setVariableDependencies(velVariable, [posVariable, velVariable, windVariable]);
    gpuCompute.setVariableDependencies(posVariable, [posVariable, velVariable]);

    // Initialize wind uniforms
    const wUni = windVariable.material.uniforms;
    wUni.uTime = { value: 0.0 };
    wUni.uDelta = { value: 0.016 };
    wUni.uWindStrength = { value: 1.0 };
    wUni.uWindDirection = { value: new THREE.Vector3(1.0, 0.0, 0.0).normalize() };
    wUni.uWindGustiness = { value: 0.5 };
    wUni.uWindFrequency = { value: 2.0 };
    wUni.uWindSpeed = { value: 2.0 };
    wUni.uCreatureVelocity = { value: new THREE.Vector3() };

    // Initialize velocity uniforms
    const vUni = velVariable.material.uniforms;
    vUni.uTime = { value: 0.0 };
    vUni.uDelta = { value: 0.016 };
    vUni.uStiffness = { value: 26.0 };
    vUni.uDamping = { value: 5.2 };
    vUni.uGravity = { value: 2.8 };
    vUni.uWindStrength = { value: 1.4 };
    vUni.uCreatureVelocity = { value: new THREE.Vector3() };
    vUni.uCreatureAngularVel = { value: new THREE.Vector3() };
    vUni.uPointerLocalPos = { value: new THREE.Vector3(0, 0, 1) };
    vUni.uPointerVelocity = { value: new THREE.Vector3() };
    vUni.uPointerActive = { value: 0.0 };
    vUni.uBoopShockwave = { value: 0.0 };

    // Initialize position uniforms
    const pUni = posVariable.material.uniforms;
    pUni.uDelta = { value: 0.016 };
    pUni.uPointerLocalPos = { value: new THREE.Vector3(0, 0, 1) };
    pUni.uPointerActive = { value: 0.0 };

    const error = gpuCompute.init();
    if (error === null) {
      isGPGPUActive = true;
    } else {
      console.warn('GPGPU fallback mode enabled:', error);
    }
  } catch (e) {
    console.warn('GPGPU initialization fell back to analytical shader physics:', e);
  }

  function step({
    time,
    dt,
    windStrength,
    windDirection,
    windGustiness = 0.5,
    windFrequency = 2.0,
    windSpeed = 2.0,
    creatureVelocity,
    creatureAngularVel,
    pointerLocalPos,
    pointerVelocity,
    pointerActive,
    boopShockwave
  }) {
    if (!isGPGPUActive || !gpuCompute) {
      return {
        posTexture: fallbackTexture,
        windTexture: fallbackWindTexture
      };
    }

    // Step 1: Update Wind Simulation uniforms
    const wUni = windVariable.material.uniforms;
    wUni.uTime.value = time;
    wUni.uDelta.value = dt;
    wUni.uWindStrength.value = windStrength;
    if (windDirection) {
      wUni.uWindDirection.value.copy(windDirection);
    }
    wUni.uWindGustiness.value = windGustiness;
    wUni.uWindFrequency.value = windFrequency;
    wUni.uWindSpeed.value = windSpeed;
    if (creatureVelocity) {
      wUni.uCreatureVelocity.value.copy(creatureVelocity);
    }

    // Step 2: Update Velocity uniforms
    const vUni = velVariable.material.uniforms;
    vUni.uTime.value = time;
    vUni.uDelta.value = dt;
    vUni.uWindStrength.value = windStrength;
    if (creatureVelocity) {
      vUni.uCreatureVelocity.value.copy(creatureVelocity);
    }
    if (creatureAngularVel) {
      vUni.uCreatureAngularVel.value.copy(creatureAngularVel);
    }
    if (pointerLocalPos) {
      vUni.uPointerLocalPos.value.copy(pointerLocalPos);
    }
    if (pointerVelocity) {
      vUni.uPointerVelocity.value.copy(pointerVelocity);
    }
    vUni.uPointerActive.value = pointerActive ? 1.0 : 0.0;
    vUni.uBoopShockwave.value = boopShockwave;

    // Step 3: Update Position uniforms
    const pUni = posVariable.material.uniforms;
    pUni.uDelta.value = dt;
    if (pointerLocalPos) {
      pUni.uPointerLocalPos.value.copy(pointerLocalPos);
    }
    pUni.uPointerActive.value = pointerActive ? 1.0 : 0.0;

    gpuCompute.compute();

    return {
      posTexture: gpuCompute.getCurrentRenderTarget(posVariable).texture,
      windTexture: gpuCompute.getCurrentRenderTarget(windVariable).texture
    };
  }

  return {
    step,
    get isGPGPUActive() {
      return isGPGPUActive;
    },
    get fallbackTexture() {
      return fallbackTexture;
    },
    get fallbackWindTexture() {
      return fallbackWindTexture;
    }
  };
}


