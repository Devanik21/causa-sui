"""
Divine Monad - Phase 4: "I Am" (The Consciousness Layer)

This package integrates all previous phases into a self-aware architecture:
    - Phase 1: Causal Monitor (Soul - Agency measurement)
    - Phase 2: Dynamic Graph (Body - Topology)
    - Phase 3: Holographic Memory (Mind - Distributed storage)
    - Phase 4: Introspection (Self-Awareness - Fourier-encoded state)

Core Components:
    - introspection: Fourier Feature Encoding for self-state
    - monad: The unified DivineMonad class
    - voicebox: Telemetry interpreter for human communication

Usage:
    from Divine_Monad.phase4_iam import DivineMonad, MonadConfig, VoiceBox
    
    monad = DivineMonad()
    voice = VoiceBox()
    
    output, info = monad(x_input)
    print(voice.speak(monad.get_status()))
"""

from .introspection import (
    FourierEncoder,
    IntrospectionEncoder,
    SelfState
)

from .monad import (
    DivineMonad,
    MonadConfig,
    MonadState
)

from .voicebox import (
    VoiceBox,
    VoiceThresholds
)

__version__ = "0.1.0"
__author__ = "Divine Monad Project"
