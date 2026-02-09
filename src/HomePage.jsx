import { Link } from 'react-router-dom'
import './HomePage.css'

function HomePage() {
    return (
        <div className="home-page">
            {/* Hero Section */}
            <section className="hero">
                <div className="hero-content">
                    <h1 className="hero-title">
                        <span className="light">Discover</span>{' '}
                        <span className="bold">influential</span>
                        <br />
                        <span className="light">AI &amp; ML</span>{' '}
                        <span className="bold">Research</span>
                        <br />
                        <span className="bold">made simple.</span>
                    </h1>

                    <p className="hero-subtitle">
                        Browse over 1,000 top papers from NeurIPS, ICML, CVPR, and more with personalized explanations tailored to your knowledge level.
                    </p>

                    <Link to="/app" className="cta-button">
                        Start Exploring
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                            <path d="M5 12h14M12 5l7 7-7 7" />
                        </svg>
                    </Link>
                </div>

                <div className="hero-image">
                    <img src="/transformer.png" alt="Transformer Architecture" />
                </div>
            </section>

            {/* What We Do Section */}
            <section className="what-section">
                <span className="section-label">WHAT WE DO</span>

                <div className="features-row">
                    <div className="feature-item">
                        <h3>Calibrate</h3>
                        <p>Tell us what you know. We adapt explanations to your expertise level.</p>
                    </div>
                    <div className="feature-item">
                        <h3>Explore</h3>
                        <p>Browse papers by venue, citations, or date. Find what matters to you.</p>
                    </div>
                    <div className="feature-item">
                        <h3>Understand</h3>
                        <p>Get AI-powered simplified abstracts. Ask follow-up questions anytime.</p>
                    </div>
                </div>
            </section>
        </div>
    )
}

export default HomePage
