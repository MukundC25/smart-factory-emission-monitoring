import { useEffect, useMemo, useRef, useState } from 'react'
import { BrowserRouter, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import mapboxgl from 'mapbox-gl'
import { gsap } from 'gsap'
import Lottie from 'lottie-react'
import { Line } from 'react-chartjs-2'
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from 'chart.js'
import 'mapbox-gl/dist/mapbox-gl.css'
import './App.css'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const MAPBOX_TOKEN =
  import.meta.env.VITE_MAPBOX_TOKEN ||
  'pk.eyJ1IjoibXVrdW5kMjAzMyIsImEiOiJjbW1xNThkdWMwcnYzMnFxdHJtNXFycmxhIn0.lfX0rAPb3cx_C7XGj-yOgw'

const CITY_OPTIONS = ['Delhi', 'Mumbai', 'Pune', 'Bangalore', 'Chennai']

const CITY_CENTERS = {
  Delhi: [77.1025, 28.7041],
  Mumbai: [72.8777, 19.076],
  Pune: [73.8567, 18.5204],
  Bangalore: [77.5946, 12.9716],
  Chennai: [80.2707, 13.0827],
}

const ROUTE_ORDER = ['/', '/city', '/questions', '/analysis', '/map', '/factory', '/ai-score', '/solutions', '/transformation']

const QUESTIONS = [
  {
    id: 'sector',
    label: 'Which industrial sector do you want to analyze?',
    options: ['Steel', 'Chemical', 'Power Plant', 'Manufacturing', 'Mixed Industry'],
  },
  {
    id: 'concern',
    label: 'What pollution concern is most important?',
    options: ['Air Quality', 'CO2 Emissions', 'Industrial Smoke', 'Water Pollution'],
  },
  {
    id: 'scope',
    label: 'What monitoring level do you want?',
    options: ['City Level', 'Industrial Zone', 'Specific Factory'],
  },
]

function normalizeScore(value) {
  const score = Number(value ?? 55)
  if (!Number.isFinite(score)) return 55
  return score <= 10 ? Math.round(score * 10) : Math.round(score)
}

function getRiskLabel(score) {
  if (score >= 70) return 'High'
  if (score >= 40) return 'Moderate'
  return 'Low'
}

function getMarkerColorClass(score) {
  if (score >= 70) return 'bad'
  if (score >= 40) return 'warn'
  return 'good'
}

function readState() {
  try {
    const raw = localStorage.getItem('sfem_ui_state')
    if (!raw) return null
    return JSON.parse(raw)
  } catch {
    return null
  }
}

function deriveFactoryMetrics(factory) {
  const score = normalizeScore(factory?.pollution_score)
  const pm25 = Number(factory?.latest_pm25 || (20 + score * 1.2).toFixed(1))
  const pm10 = Number(factory?.latest_pm10 || (34 + score * 1.55).toFixed(1))
  const no2 = Number((16 + score * 0.75).toFixed(1))
  const so2 = Number((7 + score * 0.5).toFixed(1))
  const co = Number((0.45 + score * 0.045).toFixed(2))
  const o3 = Number((14 + score * 0.43).toFixed(1))
  const trend = Array.from({ length: 7 }).map((_, index) => {
    const wobble = (Math.sin((score + index * 17) * 91.771) * 42211.43 - Math.floor(Math.sin((score + index * 17) * 91.771) * 42211.43)) * 12 - 6
    return Math.max(10, Math.round(pm25 + wobble))
  })
  return { score, pm25, pm10, no2, so2, co, o3, trend }
}

function useFactoryData(city) {
  const [loading, setLoading] = useState(true)
  const [factories, setFactories] = useState([])

  useEffect(() => {
    let active = true
    async function fetchFactories() {
      setLoading(true)
      try {
        const query = new URLSearchParams({ city, limit: '350' })
        const response = await fetch(`${API_BASE_URL}/factories?${query}`)
        if (!response.ok) {
          throw new Error('Factory fetch failed')
        }
        const payload = await response.json()
        if (!active) return
        const items = (payload.data || []).map((item, index) => {
          const score = normalizeScore(item.pollution_score)
          return {
            ...item,
            pollution_score: score,
            risk_level: item.risk_level || getRiskLabel(score),
            primary_pollutant:
              item.primary_pollutant || (score >= 70 ? 'PM2.5' : score >= 40 ? 'NO2' : 'CO'),
            latest_pm25: Number(item.latest_pm25 || (22 + score * 1.1 + index % 7).toFixed(1)),
            latest_pm10: Number(item.latest_pm10 || (36 + score * 1.4 + index % 11).toFixed(1)),
            latitude: Number(item.latitude),
            longitude: Number(item.longitude),
          }
        })
        setFactories(items)
      } catch {
        if (active) setFactories([])
      } finally {
        if (active) setLoading(false)
      }
    }
    fetchFactories()
    return () => {
      active = false
    }
  }, [city])

  return { loading, factories }
}

function usePersistentState() {
  const initial = useMemo(
    () =>
      readState() || {
        city: 'Pune',
        answers: {
          sector: 'Steel',
          concern: 'Air Quality',
          scope: 'City Level',
        },
        selectedFactoryId: null,
      },
    []
  )

  const [city, setCity] = useState(initial.city)
  const [answers, setAnswers] = useState(initial.answers)
  const [selectedFactoryId, setSelectedFactoryId] = useState(initial.selectedFactoryId)

  useEffect(() => {
    localStorage.setItem(
      'sfem_ui_state',
      JSON.stringify({
        city,
        answers,
        selectedFactoryId,
      })
    )
  }, [city, answers, selectedFactoryId])

  return { city, setCity, answers, setAnswers, selectedFactoryId, setSelectedFactoryId }
}

function useFlowNavigation() {
  const navigate = useNavigate()
  const location = useLocation()
  const currentIndex = Math.max(0, ROUTE_ORDER.indexOf(location.pathname))
  return {
    goNext() {
      navigate(ROUTE_ORDER[Math.min(ROUTE_ORDER.length - 1, currentIndex + 1)])
    },
    goBack() {
      navigate(ROUTE_ORDER[Math.max(0, currentIndex - 1)])
    },
    canGoBack: currentIndex > 0,
    canGoNext: currentIndex < ROUTE_ORDER.length - 1,
  }
}

// ============================================
// NAVIGATION ARROWS COMPONENT
// ============================================
function NavArrows({ onBack, onNext, canGoBack = true, canGoNext = true }) {
  return (
    <>
      <button 
        className="nav-arrow left" 
        onClick={onBack}
        disabled={!canGoBack}
        aria-label="Go back"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M15 18l-6-6 6-6" />
        </svg>
      </button>
      <button 
        className="nav-arrow right" 
        onClick={onNext}
        disabled={!canGoNext}
        aria-label="Go next"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M9 18l6-6-6-6" />
        </svg>
      </button>
    </>
  )
}

// ============================================
// SMOKE/FOG LOADING SCREEN WITH LOTTIE
// ============================================
function SmokeLoadingScreen({ onComplete, isClosing = false }) {
  const containerRef = useRef(null)
  const [animationData, setAnimationData] = useState(null)

  // Load smoke/fog Lottie animation with timeout
  useEffect(() => {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000)

    fetch('https://lottie.host/4b880c43-9676-4825-9dd9-e81456d38e4b/1Yx5k1i4yX.json', {
      signal: controller.signal,
    })
      .then(res => res.json())
      .then(data => setAnimationData(data))
      .catch(() => {})
      .finally(() => clearTimeout(timeoutId))

    return () => {
      clearTimeout(timeoutId)
      controller.abort()
    }
  }, [])

  useEffect(() => {
    if (!containerRef.current) return
    
    const tl = gsap.timeline()
    
    if (isClosing) {
      // Closing animation - smoke clears
      tl.to(containerRef.current, {
        opacity: 0,
        scale: 1.1,
        duration: 1.5,
        ease: 'power2.inOut',
        onComplete
      })
    } else {
      // Opening animation - smoke appears then clears
      tl.fromTo(containerRef.current, 
        { opacity: 0 },
        { opacity: 1, duration: 0.5 }
      )
      .to(containerRef.current, {
        opacity: 0,
        scale: 1.05,
        duration: 2,
        delay: 2,
        ease: 'power2.inOut',
        onComplete
      })
    }

    return () => tl.kill()
  }, [isClosing, onComplete])

  return (
    <div 
      ref={containerRef}
      className="smoke-loading-screen"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        background: 'transparent',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {/* Fullscreen fog SVG background - base layer */}
      <div 
        className="fog-bg"
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: "url('/fog.svg')",
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
          opacity: 0.6,
        }}
      />
      
      {/* Animated fog overlay with multiple layers */}
      <div 
        className="fog-layer-1"
        style={{
          position: 'absolute',
          inset: '-20%',
          backgroundImage: "url('/fog.svg')",
          backgroundSize: '150% 150%',
          backgroundPosition: 'center',
          opacity: 0.4,
          animation: 'fog-drift-1 20s ease-in-out infinite',
          filter: 'blur(10px)',
        }}
      />
      
      <div 
        className="fog-layer-2"
        style={{
          position: 'absolute',
          inset: '-10%',
          backgroundImage: "url('/fog.svg')",
          backgroundSize: '120% 120%',
          backgroundPosition: 'center',
          opacity: 0.3,
          animation: 'fog-drift-2 25s ease-in-out infinite reverse',
          filter: 'blur(20px)',
        }}
      />

      {animationData ? (
        <Lottie 
          animationData={animationData}
          loop={true}
          style={{ 
            width: 300, 
            height: 300, 
            position: 'relative',
            zIndex: 10,
          }}
        />
      ) : null}
      
      <div className="loading-text" style={{ 
        color: 'white', 
        fontSize: '1.5rem', 
        fontFamily: 'Space Grotesk, sans-serif',
        marginTop: '2rem',
        opacity: 0.9,
        position: 'relative',
        zIndex: 10,
        textShadow: '0 2px 10px rgba(0,0,0,0.5)',
      }}>
        {isClosing ? 'Clearing the air...' : 'Initializing...'}
      </div>
    </div>
  )
}

// ============================================
// LAYER 1: SVG SCENE BACKGROUND
// ============================================
function SVGScene({ isClean }) {
  return (
    <div className={`scene-container ${isClean ? 'clean' : ''}`}>
      <img
        src="/Untitled%20design.svg"
        alt="City Skyline"
        className="svg-skyline"
      />
      <div className="atmospheric-overlay" />
    </div>
  )
}

// ============================================
// LAYER 2: CINEMATIC FOG SYSTEM
// ============================================
function FogLayer({ layerIndex, isClean }) {
  const fogRef = useRef(null)
  const timelineRef = useRef(null)

  // Configuration for each fog layer (depth effect)
  const config = useMemo(() => {
    const configs = [
      { speed: 120, opacity: 0.7, yOffset: 0, scale: 1.2, blur: 0 },    // Back layer - slow, dense
      { speed: 80, opacity: 0.5, yOffset: 10, scale: 1.0, blur: 2 },    // Middle layer - medium
      { speed: 50, opacity: 0.35, yOffset: -5, scale: 0.9, blur: 4 },   // Front layer - fast, light
    ]
    return configs[layerIndex] || configs[0]
  }, [layerIndex])

  useEffect(() => {
    if (!fogRef.current) return

    const fog = fogRef.current

    // Initial fade in on page load
    gsap.set(fog, { opacity: 0 })
    gsap.to(fog, {
      opacity: config.opacity,
      duration: 2.5,
      ease: 'power2.out',
      delay: layerIndex * 0.4,
    })

    // Continuous drift animation
    timelineRef.current = gsap.timeline({ repeat: -1 })
    timelineRef.current
      .fromTo(fog,
        { x: '-30%' },
        { x: '30%', duration: config.speed, ease: 'none' }
      )
      .fromTo(fog,
        { x: '30%' },
        { x: '-30%', duration: config.speed, ease: 'none' }
      )

    return () => {
      if (timelineRef.current) {
        timelineRef.current.kill()
      }
    }
  }, [config.opacity, config.speed, layerIndex])

  // Handle clean/solution phase - fog clears sideways
  useEffect(() => {
    if (!fogRef.current) return

    const fog = fogRef.current

    if (isClean) {
      // Clear fog - move sideways and fade
      gsap.to(fog, {
        x: layerIndex % 2 === 0 ? '100%' : '-100%',
        opacity: 0,
        duration: 2,
        ease: 'power2.inOut',
      })
    } else {
      // Restore fog
      gsap.to(fog, {
        x: '0%',
        opacity: config.opacity,
        duration: 1.5,
        ease: 'power2.out',
      })
    }
  }, [isClean, config.opacity, layerIndex])

  return (
    <div
      ref={fogRef}
      className={`fog-texture layer-${layerIndex}`}
      style={{
        backgroundImage: `url('/fog.svg')`,
        backgroundSize: '200% auto',
        backgroundRepeat: 'repeat-x',
        filter: `blur(${config.blur}px)`,
        transform: `scale(${config.scale}) translateY(${config.yOffset}%)`,
      }}
    />
  )
}

function EnvironmentalEffects({ isClean }) {
  return (
    <div className={`effects-layer ${isClean ? 'clean' : ''}`}>
      {/* Atmospheric haze overlay */}
      <div className="haze-overlay" />
      
      {/* Clean environment elements (wind turbines) */}
      <div className="clean-elements">
        <div className="wind-turbine turbine-1">
          <div className="mast" />
          <div className="blades" />
        </div>
        <div className="wind-turbine turbine-2">
          <div className="mast" />
          <div className="blades" />
        </div>
      </div>
    </div>
  )
}

// ============================================
// MOTION CLOUDS LAYER - Using Lottie
// ============================================
function CloudLayer({ isClean }) {
  const [cloudData, setCloudData] = useState(null)

  useEffect(() => {
    fetch('https://lottie.host/4b880c43-9676-4825-9dd9-e81456d38e4b/1Yx5k1i4yX.json')
      .then(res => res.json())
      .then(data => setCloudData(data))
      .catch(() => {})
  }, [])

  const cloudPositions = [
    { top: '5%', left: '10%', scale: 1.2, duration: 80 },
    { top: '12%', left: '60%', scale: 0.8, duration: 60 },
    { top: '3%', left: '30%', scale: 1.0, duration: 70 },
    { top: '18%', left: '75%', scale: 0.6, duration: 50 },
    { top: '8%', left: '45%', scale: 0.9, duration: 65 },
    { top: '22%', left: '20%', scale: 0.7, duration: 55 },
  ]

  return (
    <div className={`clouds-layer ${isClean ? 'clean' : ''}`}>
      {cloudData ? (
        cloudPositions.map((pos, i) => (
          <div
            key={i}
            className="cloud-lottie"
            style={{
              position: 'absolute',
              top: pos.top,
              left: pos.left,
              width: `${200 * pos.scale}px`,
              height: `${120 * pos.scale}px`,
              animation: `cloud-drift-lottie ${pos.duration}s linear infinite`,
              animationDelay: `${-i * 10}s`,
              opacity: isClean ? 0.9 : 0.7,
            }}
          >
            <Lottie
              animationData={cloudData}
              loop={true}
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        ))
      ) : (
        cloudPositions.map((pos, i) => (
          <svg
            key={i}
            className="cloud-svg"
            viewBox="0 0 200 120"
            style={{
              position: 'absolute',
              top: pos.top,
              left: '-200px',
              width: `${180 * pos.scale}px`,
              height: 'auto',
              animation: `cloud-drift-svg ${pos.duration}s linear infinite`,
              animationDelay: `${-i * 12}s`,
              opacity: isClean ? 0.85 : 0.65,
            }}
          >
            <defs>
              <linearGradient id={`cloudGrad${i}`} x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="rgba(255,255,255,0.95)" />
                <stop offset="50%" stopColor="rgba(255,255,255,0.8)" />
                <stop offset="100%" stopColor="rgba(240,248,255,0.4)" />
              </linearGradient>
              <filter id={`cloudBlur${i}`}>
                <feGaussianBlur in="SourceGraphic" stdDeviation="2" />
              </filter>
            </defs>
            <g filter={`url(#cloudBlur${i})`}>
              <ellipse cx="100" cy="60" rx="70" ry="35" fill={`url(#cloudGrad${i})`} />
              <ellipse cx="60" cy="55" rx="45" ry="30" fill={`url(#cloudGrad${i})`} />
              <ellipse cx="140" cy="55" rx="50" ry="32" fill={`url(#cloudGrad${i})`} />
              <ellipse cx="80" cy="40" rx="35" ry="25" fill="rgba(255,255,255,0.9)" />
              <ellipse cx="120" cy="42" rx="40" ry="28" fill="rgba(255,255,255,0.85)" />
              <ellipse cx="45" cy="65" rx="25" ry="18" fill={`url(#cloudGrad${i})`} />
              <ellipse cx="155" cy="62" rx="30" ry="20" fill={`url(#cloudGrad${i})`} />
            </g>
          </svg>
        ))
      )}
    </div>
  )
}

// ============================================
// BIRDS LAYER - Realistic flying birds
// ============================================
function BirdsLayer({ isClean }) {
  const [birdData, setBirdData] = useState(null)

  useEffect(() => {
    fetch('https://lottie.host/8f311dc5-9855-4b8f-9f8e-8f8e8f8e8f8e/8f8e8f8e.json')
      .then(res => res.json())
      .then(data => setBirdData(data))
      .catch(() => {})
  }, [])

  const birdPositions = [
    { top: '8%', duration: 25, delay: 0, scale: 0.8 },
    { top: '15%', duration: 30, delay: 8, scale: 0.6 },
    { top: '6%', duration: 22, delay: 15, scale: 0.7 },
    { top: '12%', duration: 28, delay: 5, scale: 0.5 },
    { top: '18%', duration: 35, delay: 12, scale: 0.9 },
  ]

  return (
    <div className={`birds-layer ${isClean ? 'clean' : ''}`}>
      {birdData ? (
        birdPositions.map((pos, i) => (
          <div
            key={i}
            className="bird-lottie"
            style={{
              position: 'absolute',
              top: pos.top,
              left: '-100px',
              width: `${80 * pos.scale}px`,
              height: `${60 * pos.scale}px`,
              animation: `bird-fly ${pos.duration}s linear infinite`,
              animationDelay: `${pos.delay}s`,
            }}
          >
            <Lottie
              animationData={birdData}
              loop={true}
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        ))
      ) : (
        birdPositions.map((pos, i) => (
          <svg
            key={i}
            className="bird-svg"
            viewBox="0 0 100 60"
            style={{
              position: 'absolute',
              top: pos.top,
              left: '-80px',
              width: `${60 * pos.scale}px`,
              height: 'auto',
              animation: `bird-fly ${pos.duration}s linear infinite`,
              animationDelay: `${pos.delay}s`,
            }}
          >
            <defs>
              <linearGradient id={`birdGrad${i}`} x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="rgba(60,70,80,0.9)" />
                <stop offset="100%" stopColor="rgba(40,50,60,0.7)" />
              </linearGradient>
            </defs>
            <ellipse cx="50" cy="30" rx="15" ry="6" fill={`url(#birdGrad${i})`} />
            <path
              d={`M 35 28 Q 20 ${18 - (i % 3) * 3} 10 20 Q 25 25 35 28`}
              fill={`url(#birdGrad${i})`}
              opacity="0.9"
            >
              <animate
                attributeName="d"
                values={`M 35 28 Q 20 18 10 20 Q 25 25 35 28; M 35 28 Q 20 12 8 15 Q 22 22 35 28; M 35 28 Q 20 18 10 20 Q 25 25 35 28`}
                dur="0.8s"
                repeatCount="indefinite"
              />
            </path>
            <path
              d={`M 65 28 Q 80 ${18 - (i % 3) * 3} 90 20 Q 75 25 65 28`}
              fill={`url(#birdGrad${i})`}
              opacity="0.9"
            >
              <animate
                attributeName="d"
                values={`M 65 28 Q 80 18 90 20 Q 75 25 65 28; M 65 28 Q 80 12 92 15 Q 78 22 65 28; M 65 28 Q 80 18 90 20 Q 75 25 65 28`}
                dur="0.8s"
                repeatCount="indefinite"
              />
            </path>
            <path d="M 35 30 L 20 25 L 22 32 Z" fill={`url(#birdGrad${i})`} />
            <circle cx="62" cy="28" r="5" fill={`url(#birdGrad${i})`} />
            <path d="M 66 27 L 72 28 L 66 29 Z" fill="rgba(60,70,80,0.9)" />
          </svg>
        ))
      )}
    </div>
  )
}

// ============================================
// PAGE TRANSITION SMOKE
// ============================================
function SmokeTransition() {
  const smokeRef = useRef(null)
  const location = useLocation()

  useEffect(() => {
    if (smokeRef.current) {
      const smokeWave = smokeRef.current.querySelector('.smoke-wave')
      smokeWave.classList.remove('active')
      void smokeWave.offsetWidth
      smokeWave.classList.add('active')
    }
  }, [location.pathname])

  return (
    <div className="transition-smoke" ref={smokeRef}>
      <div className="smoke-wave" />
    </div>
  )
}

// ============================================
// PAGE COMPONENTS - TRANSPARENT FLOATING UI
// ============================================

function LandingPage() {
  const { goNext } = useFlowNavigation()
  
  return (
    <div className="floating-content">
      <div className="section-label">Environmental Monitoring System</div>
      <h1 className="page-title">Smart Factory Emission Monitoring</h1>
      <p className="page-subtitle">
        Discover which factories are impacting your city's air quality and explore AI-powered solutions.
      </p>
      <button className="cta-button" onClick={goNext}>
        Start Analysis
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M5 12h14M12 5l7 7-7 7" />
        </svg>
      </button>
      <NavArrows onBack={() => {}} onNext={goNext} canGoBack={false} />
    </div>
  )
}

function CitySelector({ city, setCity }) {
  const { goBack, goNext, canGoBack } = useFlowNavigation()
  
  return (
    <div className="floating-content">
      <div className="section-label">Step 1 / 7</div>
      <h1 className="page-title">Select a City</h1>
      <p className="page-subtitle">Choose a city to analyze industrial pollution patterns.</p>
      
      <div className="city-grid">
        {CITY_OPTIONS.map((cityName) => (
          <button
            key={cityName}
            className={`city-btn ${city === cityName ? 'active' : ''}`}
            onClick={() => setCity(cityName)}
          >
            {cityName}
          </button>
        ))}
      </div>
      
      <NavArrows onBack={goBack} onNext={goNext} canGoBack={canGoBack} />
    </div>
  )
}

function QuestionFlow({ answers, setAnswers }) {
  const { goBack, goNext, canGoBack } = useFlowNavigation()
  const [currentQuestion, setCurrentQuestion] = useState(0)

  const question = QUESTIONS[currentQuestion]
  
  const handleNext = () => {
    if (currentQuestion < QUESTIONS.length - 1) {
      setCurrentQuestion(currentQuestion + 1)
    } else {
      goNext()
    }
  }

  const handleBack = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1)
    } else {
      goBack()
    }
  }

  return (
    <div className="floating-content">
      <div className="section-label">Step 2 / 7 • Question {currentQuestion + 1} of {QUESTIONS.length}</div>
      <h1 className="page-title">{question.label}</h1>
      
      <div className="question-container">
        {question.options.map((option) => (
          <label key={option} className="radio-option">
            <input
              type="radio"
              name={question.id}
              checked={answers[question.id] === option}
              onChange={() => setAnswers((prev) => ({ ...prev, [question.id]: option }))}
            />
            <span>{option}</span>
          </label>
        ))}
      </div>
      
      <NavArrows onBack={handleBack} onNext={handleNext} canGoBack={true} />
    </div>
  )
}

function AnalysisLoader() {
  const { goBack, goNext, canGoBack } = useFlowNavigation()
  
  useEffect(() => {
    const timer = setTimeout(() => goNext(), 3000)
    return () => clearTimeout(timer)
  }, [goNext])

  return (
    <div className="floating-content">
      <div className="section-label">Step 3 / 7</div>
      <div className="analysis-container">
        <div className="scanner-container">
          <div className="scanner-ring" />
          <div className="scanner-ring" />
          <div className="scanner-ring" />
          <div className="scanner-line" />
        </div>
        <div className="loading-text">Analyzing industrial emission data...</div>
        <div className="loading-subtext">AI processing factory emissions and air quality metrics</div>
      </div>
      <NavArrows onBack={goBack} onNext={() => {}} canGoBack={canGoBack} canGoNext={false} />
    </div>
  )
}

function MapView({ city, factories, loading, selectedFactoryId, setSelectedFactoryId }) {
  const mapRef = useRef(null)
  const mapContainerRef = useRef(null)
  const markersRef = useRef([])
  const heatmapLayerRef = useRef(null)
  const { goBack, goNext, canGoBack } = useFlowNavigation()
  const [heatmapData, setHeatmapData] = useState([])

  // Fetch heatmap data
  useEffect(() => {
    async function fetchHeatmap() {
      try {
        const response = await fetch(`${API_BASE_URL}/pollution/heatmap/data?city=${city}&parameter=aqi_index&limit=2000`)
        if (response.ok) {
          const data = await response.json()
          setHeatmapData(data.points || [])
        }
      } catch (err) {
        console.error('Heatmap fetch failed:', err)
      }
    }
    if (city) fetchHeatmap()
  }, [city])

  const selectedFactory = useMemo(
    () => factories.find((item) => item.factory_id === selectedFactoryId) || factories[0],
    [factories, selectedFactoryId]
  )

  useEffect(() => {
    if (!factories.length || selectedFactoryId) return
    setSelectedFactoryId(factories[0].factory_id)
  }, [factories, selectedFactoryId, setSelectedFactoryId])

  useEffect(() => {
    if (!mapContainerRef.current) return
    mapboxgl.accessToken = MAPBOX_TOKEN
    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: 'mapbox://styles/mapbox/light-v11',
      center: CITY_CENTERS[city] || CITY_CENTERS.Delhi,
      zoom: 10,
    })
    map.addControl(new mapboxgl.NavigationControl(), 'top-right')
    mapRef.current = map

    // Add heatmap source and layer when map loads
    map.on('load', () => {
      map.addSource('pollution-heatmap', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: []
        }
      })

      map.addLayer({
        id: 'heatmap-layer',
        type: 'heatmap',
        source: 'pollution-heatmap',
        paint: {
          'heatmap-weight': ['get', 'intensity'],
          'heatmap-intensity': 1,
          'heatmap-color': [
            'interpolate',
            ['linear'],
            ['heatmap-density'],
            0, 'rgba(0, 255, 0, 0)',
            0.2, 'rgba(255, 255, 0, 0.5)',
            0.5, 'rgba(255, 165, 0, 0.6)',
            1, 'rgba(255, 0, 0, 0.8)'
          ],
          'heatmap-radius': 30,
          'heatmap-opacity': 0.6
        }
      })

      heatmapLayerRef.current = map.getLayer('heatmap-layer')
    })

    return () => {
      markersRef.current.forEach((marker) => marker.remove())
      markersRef.current = []
      map.remove()
    }
  }, [city])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    markersRef.current.forEach((marker) => marker.remove())
    markersRef.current = []

    factories.forEach((factory) => {
      const el = document.createElement('div')
      el.className = 'marker-container'
      const colorClass = getMarkerColorClass(factory.pollution_score)
      el.innerHTML = `<div class="marker-dot ${colorClass}"></div>`
      
      const popupHtml = `
        <div style="font-family: Inter, sans-serif;">
          <strong style="font-size: 14px; color: #2D3748;">${factory.factory_name}</strong><br/>
          <span style="font-size: 12px; color: #718096;">${factory.industry_type}</span><br/>
          <span style="font-size: 12px; margin-top: 4px; display: inline-block; padding: 2px 8px; border-radius: 4px; background: ${
            factory.pollution_score >= 70 ? 'rgba(229, 62, 62, 0.2)' : 
            factory.pollution_score >= 40 ? 'rgba(237, 137, 54, 0.2)' : 
            'rgba(72, 187, 120, 0.2)'
          }; color: ${
            factory.pollution_score >= 70 ? '#E53E3E' : 
            factory.pollution_score >= 40 ? '#ED8936' : 
            '#48BB78'
          };">Score: ${factory.pollution_score}</span>
        </div>
      `
      
      const marker = new mapboxgl.Marker(el)
        .setLngLat([factory.longitude, factory.latitude])
        .setPopup(new mapboxgl.Popup({ offset: 15 }).setHTML(popupHtml))
        .addTo(map)

      el.addEventListener('click', () => setSelectedFactoryId(factory.factory_id))
      markersRef.current.push(marker)
    })
  }, [factories, setSelectedFactoryId])

  // Update heatmap data when it changes
  useEffect(() => {
    const map = mapRef.current
    if (!map || !heatmapData.length) return

    const features = heatmapData.map(([lat, lon, intensity]) => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [lon, lat]
      },
      properties: { intensity: Math.min(intensity / 100, 1) }
    }))

    const source = map.getSource('pollution-heatmap')
    if (source) {
      source.setData({
        type: 'FeatureCollection',
        features
      })
    }
  }, [heatmapData])

  const handleViewDetails = () => {
    if (selectedFactoryId) {
      goNext()
    }
  }

  return (
    <div className="floating-content" style={{ maxWidth: '900px' }}>
      <div className="section-label">Step 4 / 7</div>
      <h1 className="page-title">Factory Map: {city}</h1>
      <p className="page-subtitle">Click markers to view details. The heatmap shows pollution intensity.</p>
      
      {loading && <p style={{ color: '#718096', marginBottom: '1rem' }}>Loading factories...</p>}
      
      {!loading && factories.length > 0 && (
        <div style={{ marginBottom: '1rem', padding: '0.75rem', background: 'rgba(255,255,255,0.5)', borderRadius: '8px' }}>
          <div style={{ fontSize: '0.875rem', color: '#4A5568' }}>
            Selected: <strong>{selectedFactory?.factory_name || 'None'}</strong>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#718096', marginTop: '0.25rem' }}>
            {selectedFactory?.industry_type} • Risk: {selectedFactory?.risk_level || 'N/A'}
          </div>
        </div>
      )}
      
      <div className="map-wrapper">
        <div ref={mapContainerRef} className="map-container" />
      </div>
      
      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginTop: '1rem' }}>
        <button 
          className="cta-button" 
          onClick={handleViewDetails}
          disabled={!selectedFactoryId}
          style={{ opacity: selectedFactoryId ? 1 : 0.5 }}
        >
          View Factory Details
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        </button>
      </div>
      
      <NavArrows onBack={goBack} onNext={goNext} canGoBack={canGoBack} />
    </div>
  )
}

function FactoryDetails({ factories, selectedFactoryId, setSelectedFactoryId }) {
  const { goBack, goNext, canGoBack } = useFlowNavigation()
  const factory = factories.find((item) => item.factory_id === selectedFactoryId) || factories[0]

  useEffect(() => {
    if (!factory && factories.length) {
      setSelectedFactoryId(factories[0].factory_id)
    }
  }, [factory, factories, setSelectedFactoryId])

  if (!factory) {
    return <Navigate to="/map" replace />
  }

  const metrics = deriveFactoryMetrics(factory)
  const colorClass = getMarkerColorClass(metrics.score)
  
  const chartData = {
    labels: ['D-6', 'D-5', 'D-4', 'D-3', 'D-2', 'D-1', 'Today'],
    datasets: [
      {
        label: 'PM2.5 Trend',
        data: metrics.trend,
        borderColor: metrics.score >= 70 ? '#E53E3E' : metrics.score >= 40 ? '#ED8936' : '#48BB78',
        backgroundColor: metrics.score >= 70 ? 'rgba(229, 62, 62, 0.2)' : metrics.score >= 40 ? 'rgba(237, 137, 54, 0.2)' : 'rgba(72, 187, 120, 0.2)',
        fill: true,
        tension: 0.35,
      },
    ],
  }

  return (
    <div className="floating-content">
      <div className="section-label">Step 5 / 7</div>
      <h1 className="page-title">{factory.factory_name}</h1>
      <p className="page-subtitle">{factory.industry_type} • Pollution Impact Report</p>
      
      <div className="badge-group">
        <span className={`badge ${colorClass}`}>Risk: {factory.risk_level || getRiskLabel(metrics.score)}</span>
        <span className="badge">Primary: {factory.primary_pollutant || 'PM2.5'}</span>
      </div>

      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-label">Pollution Score</div>
          <div className={`metric-value ${colorClass}`}>{metrics.score}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">PM2.5</div>
          <div className="metric-value">{metrics.pm25}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">PM10</div>
          <div className="metric-value">{metrics.pm10}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">NO2</div>
          <div className="metric-value">{metrics.no2}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">SO2</div>
          <div className="metric-value">{metrics.so2}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">CO</div>
          <div className="metric-value">{metrics.co}</div>
        </div>
      </div>

      <div className="chart-container">
        <Line
          data={chartData}
          options={{
            responsive: true,
            plugins: {
              legend: { labels: { color: '#2D3748' } },
            },
            scales: {
              x: { ticks: { color: '#718096' }, grid: { color: 'rgba(0,0,0,0.05)' } },
              y: { ticks: { color: '#718096' }, grid: { color: 'rgba(0,0,0,0.05)' } },
            },
          }}
        />
      </div>
      
      <NavArrows onBack={goBack} onNext={goNext} canGoBack={canGoBack} />
    </div>
  )
}

function AIScorePage({ factories, selectedFactoryId }) {
  const { goBack, goNext, canGoBack } = useFlowNavigation()
  const factory = factories.find((item) => item.factory_id === selectedFactoryId) || factories[0]
  
  if (!factory) {
    return <Navigate to="/map" replace />
  }
  
  const metrics = deriveFactoryMetrics(factory)
  const level = metrics.score >= 70 ? 'HIGH' : metrics.score >= 40 ? 'MODERATE' : 'LOW'
  const radius = (1.3 + metrics.score / 22).toFixed(1)
  const population = (0.24 + metrics.score * 0.014).toFixed(2)
  const colorClass = getMarkerColorClass(metrics.score)

  return (
    <div className="floating-content">
      <div className="section-label">Step 6 / 7</div>
      <h1 className="page-title">AI Impact Analysis</h1>
      <p className="page-subtitle">{factory.factory_name}</p>
      
      <div className="metric-grid" style={{ marginTop: '2rem' }}>
        <div className="metric-card">
          <div className="metric-label">Impact Level</div>
          <div className={`metric-value ${colorClass}`}>{level}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Impact Radius</div>
          <div className="metric-value">{radius} km</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Affected Population</div>
          <div className="metric-value">~{population}M</div>
        </div>
      </div>
      
      <NavArrows onBack={goBack} onNext={goNext} canGoBack={canGoBack} />
    </div>
  )
}

function SolutionView({ factories, selectedFactoryId }) {
  const { goBack, goNext, canGoBack } = useFlowNavigation()
  const factory = factories.find((item) => item.factory_id === selectedFactoryId) || factories[0]
  
  if (!factory) {
    return <Navigate to="/map" replace />
  }
  
  const suggestions = (factory.recommendation || '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

  const defaultSolutions = [
    { title: 'Install scrubber systems', desc: 'Advanced wet scrubbers can remove 95%+ of particulate matter' },
    { title: 'Upgrade filtration units', desc: 'HEPA and activated carbon filters for toxic gas removal' },
    { title: 'Transition to cleaner fuel', desc: 'Switch from coal to natural gas or renewable energy sources' },
    { title: 'Real-time monitoring', desc: 'IoT sensors with AI analytics for emission tracking' },
  ]

  const solutionList = suggestions.length 
    ? suggestions.map((s, i) => ({ title: s, desc: 'AI-recommended improvement' }))
    : defaultSolutions

  return (
    <div className="floating-content">
      <div className="section-label">Step 7 / 7</div>
      <h1 className="page-title">Recommended Solutions</h1>
      <p className="page-subtitle">AI-generated recommendations for {factory.factory_name}</p>
      
      <ul className="solution-list">
        {solutionList.map((solution, idx) => (
          <li key={idx} className="solution-item">
            <div className="solution-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
            </div>
            <div className="solution-text">
              <div className="solution-title">{solution.title}</div>
              <div className="solution-desc">{solution.desc}</div>
            </div>
          </li>
        ))}
      </ul>
      
      <NavArrows onBack={goBack} onNext={goNext} canGoBack={canGoBack} />
    </div>
  )
}

function TransformationPage() {
  const navigate = useNavigate()
  const { goBack, canGoBack } = useFlowNavigation()
  
  return (
    <div className="floating-content">
      <div className="section-label">Complete</div>
      <h1 className="page-title">A Cleaner Future</h1>
      <p className="page-subtitle">
        With sustainable practices, renewable energy, and AI-powered monitoring, 
        we can transform industrial zones into cleaner environments.
      </p>
      
      <div className="stats-row">
        <div className="stat-item">
          <div className="stat-value">↓45%</div>
          <div className="stat-label">Emissions</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">↑60%</div>
          <div className="stat-label">Renewable Energy</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">100%</div>
          <div className="stat-label">Transparency</div>
        </div>
      </div>
      
      <button className="cta-button" onClick={() => navigate('/')}>
        Start New Analysis
      </button>
      
      <NavArrows onBack={goBack} onNext={() => {}} canGoBack={canGoBack} canGoNext={false} />
    </div>
  )
}

// ============================================
// MAIN APP SHELL
// ============================================
function AppShell() {
  const location = useLocation()
  const { city, setCity, answers, setAnswers, selectedFactoryId, setSelectedFactoryId } = usePersistentState()
  const { loading, factories } = useFactoryData(city)
  
  const [showLoading, setShowLoading] = useState(true)
  const [showSolutionSmoke, setShowSolutionSmoke] = useState(false)

  const routeIndex = Math.max(0, ROUTE_ORDER.indexOf(location.pathname))
  const isClean = routeIndex >= 8
  
  // Handle solution page smoke animation
  useEffect(() => {
    if (location.pathname === '/solutions') {
      setShowSolutionSmoke(true)
    }
  }, [location.pathname])

  return (
    <>
      {/* Initial loading screen */}
      {showLoading && (
        <SmokeLoadingScreen onComplete={() => setShowLoading(false)} />
      )}
      
      {/* Solution screen smoke closing animation */}
      {showSolutionSmoke && (
        <SmokeLoadingScreen 
          isClosing={true}
          onComplete={() => setShowSolutionSmoke(false)} 
        />
      )}
      
      {!showLoading && (
        <>
          <SVGScene isClean={isClean} />
          <CloudLayer isClean={isClean} />
          <BirdsLayer isClean={isClean} />
          <EnvironmentalEffects isClean={isClean} />
        </>
      )}
      <SmokeTransition />
      
      <div className="ui-layer" style={{ opacity: showLoading ? 0 : 1, transition: 'opacity 0.5s ease' }}>
        {!showLoading && (
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/city" element={<CitySelector city={city} setCity={setCity} />} />
            <Route path="/questions" element={<QuestionFlow answers={answers} setAnswers={setAnswers} />} />
            <Route path="/analysis" element={<AnalysisLoader />} />
            <Route
              path="/map"
              element={
                <MapView
                  city={city}
                  factories={factories}
                  loading={loading}
                  selectedFactoryId={selectedFactoryId}
                  setSelectedFactoryId={setSelectedFactoryId}
                />
              }
            />
            <Route
              path="/factory"
              element={
                <FactoryDetails
                  factories={factories}
                  selectedFactoryId={selectedFactoryId}
                  setSelectedFactoryId={setSelectedFactoryId}
                />
              }
            />
            <Route path="/ai-score" element={<AIScorePage factories={factories} selectedFactoryId={selectedFactoryId} />} />
            <Route path="/solutions" element={<SolutionView factories={factories} selectedFactoryId={selectedFactoryId} />} />
            <Route path="/transformation" element={<TransformationPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        )}
      </div>
    </>
  )
}

function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  )
}

export default App
