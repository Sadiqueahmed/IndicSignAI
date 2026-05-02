/**
 * IndicSignAI - ISL Avatar Animation System
 * Uses character.fbx with authentic Indian Sign Language animations
 */

class ISLAvatar {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.avatar = null;
        this.mixer = null;
        this.clock = new THREE.Clock();
        this.isInitialized = false;
        this.currentAnimation = null;
        this.isAnimatingSign = false;
        this.bones = {};
        this.signQueue = [];
        this.animations = [];
        this.pending = false;
        this.flag = false;
        this.speed = 0.1;
        this.pause = 800;
        this.characters = [];  // text display queue used by animation files

        // Passed to the ISL_ANIM_* dictionary files to expose the timeline queue
        this.ref = this;

        this.init();
    }

    async init() {
        if (!this.container) {
            console.error('Avatar container not found');
            return;
        }

        // ── Replace static placeholder with a progress overlay ──
        this.removePlaceholder();
        this._showLoadingProgress(0);

        try {
            this.setupScene();
            this.setupLights();
            this.addEnvironment();

            // Start render loop IMMEDIATELY so the user sees something
            this.startRenderLoop();

            // Load character model (with LoadingManager + timeout protection)
            await this.loadCharacter();

            window.addEventListener('resize', () => this.onResize());

            this.isInitialized = true;
            this._removeLoadingProgress();
            console.log('✓ ISL Avatar initialized successfully');

        } catch (error) {
            console.error('Failed to initialize ISL avatar:', error);

            // ── FALLBACK: always show something, never infinite loading ──
            this._removeLoadingProgress();
            if (!this.avatar) {
                this.createFallbackCharacter();
            }
            // Start render loop if it wasn't started
            if (!this._renderLoopRunning) {
                this.startRenderLoop();
            }
            this.isInitialized = true;
        }
    }

    /** Remove the static HTML placeholder from the container */
    removePlaceholder() {
        if (!this.container) return;
        const placeholder = this.container.querySelector('.avatar-placeholder');
        if (placeholder) {
            placeholder.remove();
            console.log('  Placeholder removed');
        }
    }

    /** Show a loading progress overlay inside the avatar container */
    _showLoadingProgress(pct) {
        if (!this.container) return;
        let overlay = this.container.querySelector('#avatarLoadingOverlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'avatarLoadingOverlay';
            Object.assign(overlay.style, {
                position: 'absolute', top: '0', left: '0', right: '0', bottom: '0',
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                gap: '0.75rem', zIndex: '15', borderRadius: '14px',
                background: 'rgba(2, 6, 23, 0.85)', backdropFilter: 'blur(8px)'
            });
            overlay.innerHTML = `
                <i class="fa-solid fa-user-astronaut" style="font-size:2.5rem; color:#6366f1; opacity:0.6;"></i>
                <div style="font-size:0.85rem; color:#94a3b8; font-weight:600;">Loading 3D Avatar...</div>
                <div style="width:120px; height:6px; background:rgba(255,255,255,0.1); border-radius:3px; overflow:hidden;">
                    <div id="avatarProgressBar" style="width:0%; height:100%; background:linear-gradient(90deg,#6366f1,#ec4899); border-radius:3px; transition:width 0.3s;"></div>
                </div>
                <div id="avatarProgressText" style="font-size:0.7rem; color:#64748b;">0%</div>
            `;
            this.container.appendChild(overlay);
        }
        // Update progress
        const bar = overlay.querySelector('#avatarProgressBar');
        const text = overlay.querySelector('#avatarProgressText');
        if (bar) bar.style.width = Math.min(pct, 100) + '%';
        if (text) text.textContent = Math.round(pct) + '%';
    }

    /** Remove the loading progress overlay — guaranteed cleanup */
    _removeLoadingProgress() {
        if (!this.container) return;
        const overlay = this.container.querySelector('#avatarLoadingOverlay');
        if (overlay) {
            overlay.style.transition = 'opacity 0.3s';
            overlay.style.opacity = '0';
            setTimeout(() => overlay.remove(), 300);
        }
    }

    setupScene() {
        this.scene = new THREE.Scene();
        // Use transparent background to blend with CSS Glassmorphism
        this.scene.background = null;
        this.scene.fog = new THREE.FogExp2(0x020617, 0.05);

        const width = this.container.clientWidth;
        const height = this.container.clientHeight || 300;

        this.camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
        this.camera.position.set(0, 1.4, 3.5);
        this.camera.lookAt(0, 1.2, 0);

        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            powerPreference: "high-performance"
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(1); // Capped at 1.0 for massive performance boost
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;

        // Clear the container and add the WebGL canvas
        this.container.innerHTML = '';
        this.container.appendChild(this.renderer.domElement);

        // Re-add the ISL video overlay elements
        this._addVideoOverlay();
    }

    /** Re-add ISL video player + word overlay that were in the original HTML */
    _addVideoOverlay() {
        const videoEl = document.createElement('video');
        videoEl.id = 'islVideoPlayer';
        Object.assign(videoEl.style, {
            position: 'absolute', top: '0', left: '0', width: '100%', height: '100%',
            objectFit: 'contain', background: '#020617',
            display: 'none', zIndex: '10', borderRadius: '14px'
        });
        this.container.appendChild(videoEl);

        const overlay = document.createElement('div');
        overlay.id = 'videoWordOverlay';
        Object.assign(overlay.style, {
            position: 'absolute', bottom: '12px', left: '50%', transform: 'translateX(-50%)',
            background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)',
            color: '#fbbf24', fontWeight: '700', fontSize: '1.1rem',
            padding: '0.3rem 1rem', borderRadius: '50px',
            display: 'none', zIndex: '11'
        });
        this.container.appendChild(overlay);
    }

    setupLights() {
        const ambient = new THREE.AmbientLight(0xffffff, 2.0);
        this.scene.add(ambient);

        // Hemisphere Light for natural lighting
        const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.5);
        hemiLight.position.set(0, 5, 0);
        this.scene.add(hemiLight);

        // Soft Key Light (Pink/Purple)
        const keyLight = new THREE.DirectionalLight(0xd946ef, 1.5);
        keyLight.position.set(3, 5, 4);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.set(512, 512);
        keyLight.shadow.bias = -0.001;
        this.scene.add(keyLight);

        // Fill Light (Blue)
        const fillLight = new THREE.DirectionalLight(0x3b82f6, 1.0);
        fillLight.position.set(-4, 2, 2);
        this.scene.add(fillLight);

        // Intense Rim/Back Light to cut model out of dark background
        const rimLight = new THREE.SpotLight(0x8b5cf6, 4.0);
        rimLight.position.set(0, 5, -5);
        rimLight.angle = Math.PI / 4;
        rimLight.penumbra = 0.5;
        this.scene.add(rimLight);
    }

    addEnvironment() {
        const platformGeo = new THREE.CircleGeometry(2, 64);
        const platformMat = new THREE.MeshStandardMaterial({
            color: 0x020617,
            roughness: 0.2,
            metalness: 0.8
        });
        const platform = new THREE.Mesh(platformGeo, platformMat);
        platform.rotation.x = -Math.PI / 2;
        platform.receiveShadow = true;
        this.scene.add(platform);

        const grid = new THREE.GridHelper(4, 20, 0x6366f1, 0x334155);
        grid.position.y = 0.01;
        grid.material.opacity = 0.2;
        grid.material.transparent = true;
        this.scene.add(grid);
    }

    async loadCharacter() {
        // Dynamically load FBXLoader if not already present
        if (typeof THREE.FBXLoader === 'undefined') {
            try {
                // fflate is a required dependency for FBXLoader
                await this.loadScript('https://cdn.jsdelivr.net/npm/fflate@0.4.8/umd/index.js');
                await this.loadScript('https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/FBXLoader.js');
            } catch (cdnError) {
                console.error('Failed to load FBXLoader CDN scripts:', cdnError);
                this._removeLoadingProgress();
                this.createFallbackCharacter();
                return;
            }
        }

        // Race the loader against a 15-second timeout
        return Promise.race([
            this._doLoadFBX(),
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error('FBX load timeout (15s)')), 15000)
            )
        ]).catch(err => {
            console.error('FBX loading failed:', err.message);
            this._removeLoadingProgress();
            if (!this.avatar) {
                this.createFallbackCharacter();
            }
        });
    }

    _doLoadFBX() {
        return new Promise((resolve, reject) => {
            const modelPath = '/static/assets/character.fbx';

            // ── THREE.LoadingManager integration ──
            const manager = new THREE.LoadingManager();

            manager.onLoad = () => {
                console.log('✓ LoadingManager: All resources loaded');
                this._removeLoadingProgress();
            };

            manager.onProgress = (url, loaded, total) => {
                if (total > 0) {
                    const pct = (loaded / total * 100);
                    this._showLoadingProgress(pct);
                }
            };

            manager.onError = (url) => {
                console.error(`✗ LoadingManager error: Failed to load resource: ${url}`);
                this._removeLoadingProgress();
            };

            const loader = new THREE.FBXLoader(manager);

            console.log('--- Loading FBX model from:', modelPath, '---');

            // Pre-flight: check if the URL is reachable to give better error messages
            fetch(modelPath, { method: 'HEAD' }).then(resp => {
                if (!resp.ok) {
                    console.error(`✗ Pre-flight check failed: ${modelPath} returned HTTP ${resp.status} (${resp.statusText})`);
                }
            }).catch(fetchErr => {
                console.error(`✗ Pre-flight network error for ${modelPath}:`, fetchErr.message);
            });

            loader.load(
                modelPath,
                (object) => {
                    console.log('✓ character.fbx loaded successfully');

                    // Debug: count meshes and triangles
                    let meshCount = 0, triCount = 0;
                    object.traverse((child) => {
                        if (child.isMesh) {
                            meshCount++;
                            if (child.geometry) {
                                triCount += (child.geometry.index)
                                    ? child.geometry.index.count / 3
                                    : (child.geometry.attributes.position?.count || 0) / 3;
                            }
                        }
                    });
                    console.log(`  Meshes: ${meshCount}, Triangles: ~${Math.round(triCount)}`);

                    // FBXLoader gives us the object directly (not gltf.scene)
                    this.processCharacter(object, object.animations);
                    this._removeLoadingProgress();
                    resolve();
                },
                (progress) => {
                    if (progress.total > 0) {
                        const pct = (progress.loaded / progress.total * 100);
                        this._showLoadingProgress(pct);
                        if (pct < 100) {
                            console.log('Loading character: ' + pct.toFixed(0) + '%');
                        }
                    } else if (progress.loaded > 0) {
                        // total unknown — show bytes loaded
                        const mb = (progress.loaded / (1024 * 1024)).toFixed(1);
                        console.log('Loading character: ' + mb + ' MB loaded...');
                        // Animate progress bar indeterminately
                        this._showLoadingProgress(Math.min(90, progress.loaded / 300000));
                    }
                },
                (error) => {
                    console.error(`✗ FBXLoader error for ${modelPath}:`, error);
                    console.error('  Possible causes: 404 Not Found, CORS issue, or corrupted FBX file');
                    console.error('  Full URL attempted:', window.location.origin + modelPath);
                    this._removeLoadingProgress();
                    reject(error);
                }
            );
        });
    }

    processCharacter(object, animations) {
        this.avatar = object;

        // Robustly normalize the character's height to exactly 1.8 units (meters)
        let box = new THREE.Box3().setFromObject(object);
        let size = box.getSize(new THREE.Vector3());
        
        console.log('--- FBX LOAD SUCCESS ---');
        console.log('Original BoundingBox Size:', size);

        // Prevent division by zero if bounds are weird
        if (size.y > 0.001) {
            const desiredHeight = 1.8;
            let scaleFactor = desiredHeight / size.y;
            
            // Sanity Check: If the native bounding box is astronomically corrupted, force standard scale
            if (scaleFactor < 0.0001 || scaleFactor > 1000) {
                console.warn('Native Bounding Box is severely corrupted. Forcing Standard Scale 1.0');
                scaleFactor = 1.0;
            }
            
            console.log('Calculated Scale Factor to achieve 1.8m:', scaleFactor);
            
            object.scale.set(scaleFactor, scaleFactor, scaleFactor);
            object.updateMatrixWorld(true);
            
            // Recompute box after scale to anchor to floor
            box = new THREE.Box3().setFromObject(object);
            console.log('New scaled Y bounds -> Min:', box.min.y, 'Max:', box.max.y);
            object.position.y = -box.min.y;
        } else {
            console.warn('Invalid Bounding Box Height. Forcing fallback scale 1.0');
            object.scale.set(1, 1, 1);
            object.position.set(0, 0, 0);
        }

        object.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;
                if (child.material) {
                    child.material.roughness = 0.4;
                    child.material.metalness = 0.2;
                    child.material.envMapIntensity = 1.0;
                }
            }
            if (child.isBone) {
                // Store bone with its ORIGINAL name (e.g. 'mixamorigLeftHand')
                this.bones[child.name] = child;

                // Create fallback aliases to match the JS animation dictionary standard
                const baseName = child.name.replace(/mixamorig[0-9]*:/, 'mixamorig');
                if (baseName !== child.name) {
                    this.bones[baseName] = child;
                }

                // Also store a stripped version for ultimate flexibility
                let strippedName = child.name.replace(/^mixamorig:?|^mixamorig[0-9]*:|^m_?/i, '');
                if (strippedName === 'Spine1') strippedName = 'Spine';
                if (strippedName === 'HeadTop_End') strippedName = 'Head';
                this.bones[strippedName] = child;
            }
        });

        this.mixer = new THREE.AnimationMixer(object);
        this.scene.add(object);

        console.log('Character processed. Bones found:', Object.keys(this.bones).length);
        console.log('Bone names:', Object.keys(this.bones).join(', '));

        if (animations && animations.length > 0) {
            console.log("Playing embedded FBX Idle Animation to break T-Pose");
            const action = this.mixer.clipAction(animations[0]);
            action.play();
            this.idleAction = action;
        } else {
            // ── IMMEDIATE T-Pose Correction ──
            console.log('No embedded animation — applying IMMEDIATE natural pose...');
            this.applyNaturalPose();

            // Safety timeout: re-apply natural pose after a short delay in case
            // skeleton wasn't fully ready on the first call
            setTimeout(() => {
                if (this.avatar && !this.idleAction) {
                    console.log('  Safety re-applying natural pose (300ms timeout)');
                    this.applyNaturalPose();
                }
            }, 300);
        }
    }

    createFallbackCharacter() {
        console.log('Creating geometric fallback character...');
        const group = new THREE.Group();

        const skinMat = new THREE.MeshStandardMaterial({ color: 0xf5d0c5, roughness: 0.5 });
        const shirtMat = new THREE.MeshStandardMaterial({ color: 0x6366f1, roughness: 0.8 });
        const pantsMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.8 });

        const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.25, 0.6, 4, 8), shirtMat);
        torso.position.y = 1.1;
        torso.castShadow = true;
        group.add(torso);

        const head = new THREE.Mesh(new THREE.SphereGeometry(0.18, 32, 32), skinMat);
        head.position.y = 1.65;
        head.castShadow = true;
        group.add(head);

        const armGeo = new THREE.CapsuleGeometry(0.08, 0.5, 4, 8);

        const leftArm = new THREE.Mesh(armGeo, skinMat);
        leftArm.position.set(-0.35, 1.3, 0);
        leftArm.rotation.z = 0.2;
        leftArm.castShadow = true;
        group.add(leftArm);
        this.bones['LeftArm'] = leftArm;

        const rightArm = new THREE.Mesh(armGeo, skinMat);
        rightArm.position.set(0.35, 1.3, 0);
        rightArm.rotation.z = -0.2;
        rightArm.castShadow = true;
        group.add(rightArm);
        this.bones['RightArm'] = rightArm;

        const leftHand = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.12, 0.05), skinMat);
        leftHand.position.set(0, -0.35, 0);
        leftArm.add(leftHand);
        this.bones['LeftHand'] = leftHand;

        const rightHand = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.12, 0.05), skinMat);
        rightHand.position.set(0, -0.35, 0);
        rightArm.add(rightHand);
        this.bones['RightHand'] = rightHand;

        const legGeo = new THREE.CapsuleGeometry(0.1, 0.7, 4, 8);

        const leftLeg = new THREE.Mesh(legGeo, pantsMat);
        leftLeg.position.set(-0.12, 0.4, 0);
        leftLeg.castShadow = true;
        group.add(leftLeg);

        const rightLeg = new THREE.Mesh(legGeo, pantsMat);
        rightLeg.position.set(0.12, 0.4, 0);
        rightLeg.castShadow = true;
        group.add(rightLeg);

        this.avatar = group;
        this.ref.avatar = this.avatar;
        this.scene.add(group);

        console.log('✓ Fallback character created and added to scene');
    }

    /**
     * IMMEDIATE T-Pose correction — directly sets bone rotations
     * without going through the animation queue pipeline.
     * Arms rotate ~65° downward for a natural standing pose.
     */
    applyNaturalPose() {
        const deg65 = 65 * (Math.PI / 180);  // 1.134 radians

        // Helper: try multiple bone name variants
        const getBone = (names) => {
            for (const n of names) {
                if (this.bones[n]) return this.bones[n];
            }
            return null;
        };

        // Left arm down by 65°
        const leftArm = getBone(['mixamorigLeftArm', 'mixamorig:LeftArm', 'LeftArm']);
        if (leftArm) {
            leftArm.rotation.z = -deg65;
            console.log('  ✓ LeftArm rotated -65° on Z');
        }

        // Left forearm natural bend
        const leftForeArm = getBone(['mixamorigLeftForeArm', 'mixamorig:LeftForeArm', 'LeftForeArm']);
        if (leftForeArm) {
            leftForeArm.rotation.y = -Math.PI / 1.5;
            console.log('  ✓ LeftForeArm rotated');
        }

        // Right arm down by 65°
        const rightArm = getBone(['mixamorigRightArm', 'mixamorig:RightArm', 'RightArm']);
        if (rightArm) {
            rightArm.rotation.z = deg65;
            console.log('  ✓ RightArm rotated +65° on Z');
        }

        // Right forearm natural bend
        const rightForeArm = getBone(['mixamorigRightForeArm', 'mixamorig:RightForeArm', 'RightForeArm']);
        if (rightForeArm) {
            rightForeArm.rotation.y = Math.PI / 1.5;
            console.log('  ✓ RightForeArm rotated');
        }

        // Slight neck tilt for natural look
        const neck = getBone(['mixamorigNeck', 'mixamorig:Neck', 'Neck']);
        if (neck) {
            neck.rotation.x = Math.PI / 12;
            console.log('  ✓ Neck tilted');
        }

        console.log('✓ Natural pose applied immediately');
    }

    playSign(text) {
        if (!this.avatar) {
            console.warn('Avatar not loaded');
            return;
        }

        if (!text || text.trim() === '') return;

        this.showSignInfo(text, "Generating ISL Sequence...");

        const words = text.trim().toUpperCase().split(/\s+/);

        for (let word of words) {
            // Check if full word function exists inside 'words.js' definitions
            const wordFunc = window[`ISL_ANIM_${word}`];

            if (wordFunc) {
                this.animations.push(['add-text', word + ' ']);
                wordFunc(this.ref);
            } else {
                // Otherwise spell out letter by letter using A-Z animations
                for (let i = 0; i < word.length; i++) {
                    const char = word[i];
                    if (i === word.length - 1) {
                        this.animations.push(['add-text', char + ' ']);
                    } else {
                        this.animations.push(['add-text', char]);
                    }

                    const charFunc = window[`ISL_ANIM_${char}`];
                    if (charFunc) {
                        charFunc(this.ref);
                    } else {
                        console.warn(`No animation mapping for character: ${char}`);
                    }
                }
            }
        }

        // Execute the pipeline if not already running
        if (this.pending === false && this.animations.length > 0) {
            this.pending = true;
            this.animate();
        }
    }

    showSignInfo(name, description) {
        const infoPanel = document.getElementById('avatarSignInfo');
        if (infoPanel) {
            infoPanel.innerHTML =
                '<div style="' +
                'background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(236, 72, 153, 0.1));' +
                'border: 1px solid rgba(99, 102, 241, 0.3);' +
                'border-radius: 12px;' +
                'padding: 1rem;' +
                'margin-top: 1rem;' +
                'animation: fadeIn 0.3s ease;' +
                '">' +
                '<div style="font-weight: 700; color: #6366f1; font-size: 1.1rem; margin-bottom: 0.5rem;">' +
                name +
                '</div>' +
                '<div style="font-size: 0.875rem; color: #94a3b8; line-height: 1.5;">' +
                description +
                '</div>' +
                '</div>';
        }
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    // ========== CONTINUOUS RENDER LOOP (always runs) ==========
    startRenderLoop() {
        if (this._renderLoopRunning) return; // prevent duplicates
        this._renderLoopRunning = true;

        const loop = () => {
            requestAnimationFrame(loop);

            const delta = this.clock.getDelta();

            // Handle Embedded Animation blending
            if (this.idleAction) {
                if (this.isAnimatingSign) {
                    this.idleAction.weight = THREE.MathUtils.lerp(this.idleAction.weight, 0, 0.1);
                } else {
                    this.idleAction.weight = THREE.MathUtils.lerp(this.idleAction.weight, 1, 0.05);
                }
            }

            // Fallback breathing if no embedded animation exists
            if (this.avatar && !this.isAnimatingSign && !this.idleAction) {
                if (this.avatar.userData.baseY === undefined) {
                    this.avatar.userData.baseY = this.avatar.position.y;
                }
                const t = performance.now() / 1000;
                this.avatar.position.y = this.avatar.userData.baseY + Math.sin(t * 1.5) * 0.005;
            }

            if (this.mixer) this.mixer.update(delta);
            if (this.renderer && this.scene && this.camera) {
                this.renderer.render(this.scene, this.camera);
            }
        };
        loop();
    }

    // ========== SIGN ANIMATION PROCESSOR ==========
    animate() {
        if (!this.avatar || this.animations.length === 0) {
            this.pending = false;
            return;
        }

        this.isAnimatingSign = true;

        requestAnimationFrame(() => this.animate());

        // Process Sign-Kit array matrices
        if (this.animations[0] && this.animations[0].length) {
            if (!this.flag) {
                // If the block is a text update cue
                if (this.animations[0][0] === 'add-text') {
                    const charAdded = this.animations[0][1];
                    console.log("Signing:", charAdded);
                    this.animations.shift();
                } else {
                    // It is a mathematical bone array block
                    for (let i = 0; i < this.animations[0].length;) {
                        let [boneName, action, axis, limit, sign] = this.animations[0][i];

                        let targetBone = this.bones[boneName];

                        if (targetBone) {
                            if (sign === "+" && targetBone[action][axis] < limit) {
                                targetBone[action][axis] += this.speed;
                                targetBone[action][axis] = Math.min(targetBone[action][axis], limit);
                                i++;
                            } else if (sign === "-" && targetBone[action][axis] > limit) {
                                targetBone[action][axis] -= this.speed;
                                targetBone[action][axis] = Math.max(targetBone[action][axis], limit);
                                i++;
                            } else {
                                this.animations[0].splice(i, 1);
                            }
                        } else {
                            // Bone not found — skip to prevent stalling
                            console.warn('Bone not found:', boneName);
                            this.animations[0].splice(i, 1);
                        }
                    }
                }
            }
        } else {
            // Pause between frames/letters
            this.flag = true;
            setTimeout(() => {
                this.flag = false;
            }, this.pause);
            this.animations.shift();
        }

        // When queue is exhausted, mark done
        if (this.animations.length === 0) {
            this.isAnimatingSign = false;
            this.pending = false;
        }
    }

    playIdle() {
        // Idle is handled by the continuous render loop breathing animation
        console.log('Avatar idle mode active');
    }

    onResize() {
        if (!this.container || !this.camera || !this.renderer) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight || 300;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    showFallback() {
        if (!this.container) return;
        this.container.innerHTML =
            '<div style="' +
            'display: flex; flex-direction: column; align-items: center; justify-content: center; ' +
            'height: 100%; color: #94a3b8; text-align: center; padding: 2rem;' +
            '">' +
            '<i class="fa-solid fa-user-astronaut" style="font-size: 3rem; margin-bottom: 1rem; color: #6366f1;"></i>' +
            '<div style="font-size: 1rem; margin-bottom: 0.5rem;">3D Avatar</div>' +
            '<div style="font-size: 0.875rem; opacity: 0.7;">Sign: ' + (this.currentSign || 'None') + '</div>' +
            '</div>';
    }
}

// Legacy compatibility
window.SignLanguageAvatar = ISLAvatar;
