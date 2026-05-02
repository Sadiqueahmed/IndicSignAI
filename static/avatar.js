/**
 * IndicSignAI - 3D Avatar for Sign Language Animation
 * Uses Three.js to display and animate the FBX character
 */

class SignLanguageAvatar {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.avatar = null;
        this.mixer = null;
        this.clock = new THREE.Clock();
        this.isInitialized = false;
        
        // Sign animations mapping (simplified keyframe animations)
        this.signAnimations = {
            'HELLO': { duration: 2000, description: 'Wave hand near face' },
            'THANK': { duration: 1500, description: 'Touch chin and move forward' },
            'YES': { duration: 1000, description: 'Nod fist up and down' },
            'NO': { duration: 1000, description: 'Shake head side to side' },
            'PLEASE': { duration: 1500, description: 'Rub palm on chest' },
            'SORRY': { duration: 1500, description: 'Fist rubs chest' },
            'GOOD': { duration: 1200, description: 'Hand from chin outward' },
            'HELP': { duration: 1500, description: 'Thumbs up on palm' },
            'LOVE': { duration: 2000, description: 'Cross arms over chest' },
            'FRIEND': { duration: 1500, description: 'Hook index fingers' },
            'EAT': { duration: 1500, description: 'Hand to mouth' },
            'DRINK': { duration: 1500, description: 'Hand tilts to mouth' },
            'WATER': { duration: 1200, description: 'W hand taps chin' },
            'FOOD': { duration: 1500, description: 'Fingers touch mouth' },
            'HOME': { duration: 1500, description: 'H hand to cheek' },
            'WORK': { duration: 1500, description: 'W hands tap' },
            'SCHOOL': { duration: 1500, description: 'S hands tap' },
            'HAPPY': { duration: 2000, description: 'Hands brush chest' },
            'SAD': { duration: 2000, description: 'Hands down face' },
            'ANGRY': { duration: 1500, description: 'Claw hands at chest' }
        };
        
        this.init();
    }
    
    async init() {
        if (!this.container) {
            console.error('Avatar container not found');
            return;
        }
        
        try {
            // Create scene
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0x0f172a);
            
            // Add fog for depth
            this.scene.fog = new THREE.Fog(0x0f172a, 10, 50);
            
            // Create camera
            const width = this.container.clientWidth;
            const height = this.container.clientHeight || 400;
            this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            this.camera.position.set(0, 1.5, 4);
            this.camera.lookAt(0, 1, 0);
            
            // Create renderer
            this.renderer = new THREE.WebGLRenderer({ 
                antialias: true, 
                alpha: true,
                powerPreference: "high-performance"
            });
            this.renderer.setSize(width, height);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            this.renderer.outputColorSpace = THREE.SRGBColorSpace;
            this.container.appendChild(this.renderer.domElement);
            
            // Add lights
            this.setupLights();
            
            // Add ground
            this.addGround();
            
            // Load avatar
            await this.loadAvatar();
            
            // Start animation loop
            this.animate();
            
            // Handle resize
            window.addEventListener('resize', () => this.onResize());
            
            this.isInitialized = true;
            console.log('✓ Sign Language Avatar initialized');
            
        } catch (error) {
            console.error('Failed to initialize avatar:', error);
            this.showFallback();
        }
    }
    
    setupLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x6366f1, 0.4);
        this.scene.add(ambientLight);
        
        // Main directional light
        const mainLight = new THREE.DirectionalLight(0xffffff, 1);
        mainLight.position.set(5, 10, 7);
        mainLight.castShadow = true;
        mainLight.shadow.mapSize.width = 2048;
        mainLight.shadow.mapSize.height = 2048;
        mainLight.shadow.camera.near = 0.1;
        mainLight.shadow.camera.far = 50;
        this.scene.add(mainLight);
        
        // Rim light for edge definition
        const rimLight = new THREE.SpotLight(0xec4899, 2);
        rimLight.position.set(-5, 5, -5);
        rimLight.lookAt(0, 1, 0);
        this.scene.add(rimLight);
        
        // Fill light
        const fillLight = new THREE.PointLight(0x06b6d4, 0.5);
        fillLight.position.set(-3, 2, 3);
        this.scene.add(fillLight);
    }
    
    addGround() {
        // Create a circular platform
        const geometry = new THREE.CircleGeometry(3, 64);
        const material = new THREE.MeshStandardMaterial({ 
            color: 0x1e293b,
            roughness: 0.8,
            metalness: 0.2
        });
        const ground = new THREE.Mesh(geometry, material);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);
        
        // Add grid helper
        const gridHelper = new THREE.GridHelper(6, 20, 0x6366f1, 0x334155);
        gridHelper.position.y = 0.01;
        gridHelper.material.opacity = 0.3;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);
    }
    
    async loadAvatar() {
        return new Promise((resolve, reject) => {
            // Check if Three.js FBXLoader is available
            if (typeof THREE.FBXLoader === 'undefined') {
                // Load FBXLoader dynamically
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/loaders/FBXLoader.js';
                script.onload = () => this.loadFBXModel(resolve, reject);
                script.onerror = () => reject(new Error('Failed to load FBXLoader'));
                document.head.appendChild(script);
            } else {
                this.loadFBXModel(resolve, reject);
            }
        });
    }
    
    loadFBXModel(resolve, reject) {
        const loader = new THREE.FBXLoader();
        const modelPath = '../assets/characters/character.fbx';
        
        loader.load(
            modelPath,
            (object) => {
                console.log('✓ FBX model loaded');
                this.avatar = object;
                
                // Scale and position
                object.scale.set(0.01, 0.01, 0.01);
                object.position.set(0, 0, 0);
                
                // Enable shadows
                object.traverse((child) => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                        
                        // Improve materials
                        if (child.material) {
                            child.material.roughness = 0.7;
                            child.material.metalness = 0.1;
                        }
                    }
                });
                
                // Setup animation mixer
                this.mixer = new THREE.AnimationMixer(object);
                
                // Add idle animation
                this.playIdleAnimation();
                
                this.scene.add(object);
                resolve();
            },
            (progress) => {
                const percent = (progress.loaded / progress.total * 100).toFixed(0);
                console.log(`Loading avatar: ${percent}%`);
            },
            (error) => {
                console.error('Error loading FBX:', error);
                this.createFallbackAvatar();
                resolve();
            }
        );
    }
    
    createFallbackAvatar() {
        // Create a simple humanoid figure as fallback
        const group = new THREE.Group();
        
        // Body
        const bodyGeo = new THREE.CapsuleGeometry(0.3, 1, 4, 8);
        const bodyMat = new THREE.MeshStandardMaterial({ color: 0x6366f1 });
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        body.position.y = 1;
        body.castShadow = true;
        group.add(body);
        
        // Head
        const headGeo = new THREE.SphereGeometry(0.25, 32, 32);
        const headMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc });
        const head = new THREE.Mesh(headGeo, headMat);
        head.position.y = 1.8;
        head.castShadow = true;
        group.add(head);
        
        // Arms
        const armGeo = new THREE.CapsuleGeometry(0.08, 0.6, 4, 8);
        const armMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc });
        
        const leftArm = new THREE.Mesh(armGeo, armMat);
        leftArm.position.set(-0.4, 1.3, 0);
        leftArm.rotation.z = 0.3;
        leftArm.castShadow = true;
        group.add(leftArm);
        
        const rightArm = new THREE.Mesh(armGeo, armMat);
        rightArm.position.set(0.4, 1.3, 0);
        rightArm.rotation.z = -0.3;
        rightArm.castShadow = true;
        group.add(rightArm);
        
        // Legs
        const legGeo = new THREE.CapsuleGeometry(0.1, 0.8, 4, 8);
        const legMat = new THREE.MeshStandardMaterial({ color: 0x1e293b });
        
        const leftLeg = new THREE.Mesh(legGeo, legMat);
        leftLeg.position.set(-0.15, 0.4, 0);
        leftLeg.castShadow = true;
        group.add(leftLeg);
        
        const rightLeg = new THREE.Mesh(legGeo, legMat);
        rightLeg.position.set(0.15, 0.4, 0);
        rightLeg.castShadow = true;
        group.add(rightLeg);
        
        this.avatar = group;
        this.scene.add(group);
        
        console.log('✓ Fallback avatar created');
    }
    
    showFallback() {
        this.container.innerHTML = `
            <div style="
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100%;
                color: #94a3b8;
                text-align: center;
                padding: 2rem;
            ">
                <i class="fa-solid fa-user-astronaut" style="font-size: 4rem; margin-bottom: 1rem; color: #6366f1;"></i>
                <p>3D Avatar preview unavailable</p>
                <p style="font-size: 0.875rem; margin-top: 0.5rem;">Sign: ${this.currentSign || 'None'}</p>
            </div>
        `;
    }
    
    playIdleAnimation() {
        if (!this.avatar) return;
        
        // Simple idle breathing animation
        const duration = 2000;
        const startY = this.avatar.position.y;
        
        const animate = (time) => {
            if (!this.isAnimatingSign) {
                const offset = Math.sin(time / duration * Math.PI * 2) * 0.02;
                this.avatar.position.y = startY + offset;
                
                // Subtle arm sway
                this.avatar.traverse((child) => {
                    if (child.name && child.name.includes('Arm')) {
                        child.rotation.z += Math.sin(time / 1000) * 0.001;
                    }
                });
            }
            requestAnimationFrame(animate);
        };
        
        requestAnimationFrame(animate);
    }
    
    animateSign(signName) {
        if (!this.avatar) {
            console.warn('Avatar not loaded');
            return;
        }
        
        const sign = this.signAnimations[signName.toUpperCase()];
        if (!sign) {
            console.warn(`No animation defined for sign: ${signName}`);
            this.showSignDescription(signName, 'Animation not available');
            return;
        }
        
        this.isAnimatingSign = true;
        this.currentSign = signName;
        
        // Show sign description
        this.showSignDescription(signName, sign.description);
        
        // Perform animation based on sign type
        this.performSignAnimation(signName);
        
        // Reset after animation
        setTimeout(() => {
            this.isAnimatingSign = false;
        }, sign.duration);
    }
    
    performSignAnimation(signName) {
        if (!this.avatar) return;
        
        const upperSign = signName.toUpperCase();
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const sign = this.signAnimations[upperSign];
            
            if (!sign || elapsed > sign.duration) {
                this.isAnimatingSign = false;
                return;
            }
            
            const progress = elapsed / sign.duration;
            
            // Different animations based on sign
            switch(upperSign) {
                case 'HELLO':
                    // Wave motion
                    this.avatar.rotation.y = Math.sin(progress * Math.PI * 4) * 0.3;
                    break;
                case 'THANK':
                    // Bow motion
                    this.avatar.rotation.x = Math.sin(progress * Math.PI) * 0.2;
                    break;
                case 'YES':
                    // Nod
                    this.avatar.rotation.x = Math.abs(Math.sin(progress * Math.PI * 2)) * 0.15;
                    break;
                case 'NO':
                    // Head shake
                    this.avatar.rotation.y = Math.sin(progress * Math.PI * 4) * 0.2;
                    break;
                case 'PLEASE':
                case 'SORRY':
                    // Circular motion
                    this.avatar.position.x = Math.sin(progress * Math.PI * 2) * 0.1;
                    break;
                default:
                    // Generic pulse
                    const scale = 1 + Math.sin(progress * Math.PI * 2) * 0.05;
                    this.avatar.scale.setScalar(scale * 0.01);
            }
            
            if (this.isAnimatingSign) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    showSignDescription(sign, description) {
        // Update UI with sign info
        const infoPanel = document.getElementById('avatarSignInfo');
        if (infoPanel) {
            infoPanel.innerHTML = `
                <div style="
                    background: rgba(99, 102, 241, 0.1);
                    border: 1px solid rgba(99, 102, 241, 0.3);
                    border-radius: 12px;
                    padding: 1rem;
                    margin-top: 1rem;
                ">
                    <div style="font-weight: 700; color: #6366f1; margin-bottom: 0.25rem;">
                        ${sign}
                    </div>
                    <div style="font-size: 0.875rem; color: #94a3b8;">
                        ${description}
                    </div>
                </div>
            `;
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        const delta = this.clock.getDelta();
        
        // Update mixer
        if (this.mixer) {
            this.mixer.update(delta);
        }
        
        // Render
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    onResize() {
        if (!this.container || !this.camera || !this.renderer) return;
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight || 400;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    // Public API
    showSign(signName) {
        this.animateSign(signName);
    }
    
    setExpression(expression) {
        // Change avatar expression/material
        if (!this.avatar) return;
        
        const colors = {
            'happy': 0xfcd34d,
            'sad': 0x60a5fa,
            'neutral': 0xf8fafc,
            'excited': 0xf472b6
        };
        
        const color = colors[expression] || colors.neutral;
        
        this.avatar.traverse((child) => {
            if (child.isMesh && child.name.includes('Head')) {
                child.material.color.setHex(color);
            }
        });
    }
    
    destroy() {
        if (this.renderer) {
            this.renderer.dispose();
            this.container.removeChild(this.renderer.domElement);
        }
        
        window.removeEventListener('resize', () => this.onResize());
    }
}

// Initialize when DOM is ready
let avatarInstance = null;

function initAvatar() {
    if (document.getElementById('avatarContainer')) {
        avatarInstance = new SignLanguageAvatar('avatarContainer');
    }
}

// Auto-initialize if container exists
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAvatar);
} else {
    initAvatar();
}

// Export for global access
window.SignLanguageAvatar = SignLanguageAvatar;
window.getAvatar = () => avatarInstance;
