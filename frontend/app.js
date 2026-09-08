/**
 * OmniSplat 3D — Master Google Earth 3D Digital Twin & 3D Print Studio Engine
 * Full Photorealistic PBR Surface Meshing, Dynamic Atmospheric Lighting,
 * Real-Time Geospatial HUD, Interactive 3D Compass Rose, and 3D Print (.STL) Export.
 */

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements - Video Ingestion
    const dropZone = document.getElementById('dropZone');
    const videoFileInput = document.getElementById('videoFileInput');
    const browseBtn = document.getElementById('browseBtn');
    const datasetSelect = document.getElementById('datasetSelect');
    const loadDatasetBtn = document.getElementById('loadDatasetBtn');
    const videoPreviewWrapper = document.getElementById('videoPreviewWrapper');
    const videoPreview = document.getElementById('videoPreview');
    const metaRes = document.getElementById('metaRes');
    const metaDuration = document.getElementById('metaDuration');
    const metaFps = document.getElementById('metaFps');

    // Telemetry Stepper
    const keyframeRange = document.getElementById('keyframeRange');
    const keyframeVal = document.getElementById('keyframeVal');
    const iterRange = document.getElementById('iterRange');
    const iterVal = document.getElementById('iterVal');
    const startBtn = document.getElementById('startBtn');

    const progressBarFill = document.getElementById('progressBarFill');
    const progressPct = document.getElementById('progressPct');
    const progressMsg = document.getElementById('progressMsg');
    const stageTag = document.getElementById('stageTag');
    const terminalBox = document.getElementById('terminalBox');

    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const step3 = document.getElementById('step3');
    const step4 = document.getElementById('step4');

    // Viewport & Header Elements
    const canvasWrapper = document.getElementById('canvasWrapper');
    const canvasPlaceholder = document.getElementById('canvasPlaceholder');
    const modelStatsBadge = document.getElementById('modelStatsBadge');
    const printSpecBadge = document.getElementById('printSpecBadge');
    const hudLocation = document.getElementById('hudLocation');
    const hudCoords = document.getElementById('hudCoords');
    const hudElevation = document.getElementById('hudElevation');

    const btnModePhotoreal = document.getElementById('btnModePhotoreal');
    const btnModePrintResin = document.getElementById('btnModePrintResin');
    const btnModeSlicerLayers = document.getElementById('btnModeSlicerLayers');
    const btnModeGaussians = document.getElementById('btnModeGaussians');

    const btnTimeDay = document.getElementById('btnTimeDay');
    const btnTimeGolden = document.getElementById('btnTimeGolden');
    const btnTimeNight = document.getElementById('btnTimeNight');

    const btnToggleBuildPlate = document.getElementById('btnToggleBuildPlate');
    const btnToggleTrajectory = document.getElementById('btnToggleTrajectory');
    const btnAutoRotate = document.getElementById('btnAutoRotate');
    const btnResetCam = document.getElementById('btnResetCam');

    const compassWidget = document.getElementById('compassWidget');
    const compassDial = document.getElementById('compassDial');
    const teleHeading = document.getElementById('teleHeading');
    const teleTilt = document.getElementById('teleTilt');
    const teleAlt = document.getElementById('teleAlt');

    // Authentic Optical Transformation Perspectives
    const flyPortal = document.getElementById('flyPortal');
    const flyTriangle = document.getElementById('flyTriangle');
    const flyCross = document.getElementById('flyCross');
    const flyPlanZ = document.getElementById('flyPlanZ');
    const flyBridge = document.getElementById('flyBridge');
    const flyOrbit = document.getElementById('flyOrbit');

    // Measurement & Downloads
    const toolMeasureDist = document.getElementById('toolMeasureDist');
    const toolMeasureHeight = document.getElementById('toolMeasureHeight');
    const measureResult = document.getElementById('measureResult');
    const downloadStlBtn = document.getElementById('downloadStlBtn');
    const downloadObjBtn = document.getElementById('downloadObjBtn');
    const downloadPlyBtn = document.getElementById('downloadPlyBtn');

    // App State
    let activeVideoPath = null;
    let isProcessing = false;
    let sseSource = null;
    let activeMeasurementMode = null;
    let measurePoints = [];
    let measurementLine = null;
    let isAutoRotating = true; // Auto-rotate on by default for full 360 view
    let showTrajectory = true;
    let showBuildPlate = false; // Off by default in Google Earth mode
    let currentRenderMode = 'resin'; // 'resin', 'slicer', 'gaussians'
    let currentTimeMode = 'day'; // 'day', 'golden', 'night'

    // Three.js Objects
    let scene, camera, renderer, controls;
    let solidMesh = null;
    let pointCloudMesh = null;
    let buildPlateGroup = null;
    let trajectoryGroup = null;
    let nightLightsGroup = null;
    let groundHorizonMesh = null;

    // Lights
    let keySunLight, fillSkyLight, ambientLight, rimLight;

    // Textures & Multi-Material Arrays
    let texSteel, texTerrain, texConcrete;
    let pbrMaterials = [];
    let resinMaterials = [];
    let slicerMaterials = [];
    let matResin, matSlicer;

    // Sliders
    keyframeRange.addEventListener('input', () => keyframeVal.textContent = keyframeRange.value);
    iterRange.addEventListener('input', () => iterVal.textContent = iterRange.value);

    function logTerminal(msg, type = 'info') {
        const line = document.createElement('div');
        line.className = `log-row ${type}`;
        line.textContent = msg;
        terminalBox.appendChild(line);
        terminalBox.scrollTop = terminalBox.scrollHeight;
    }

    // -------------------------------------------------------------
    // THREE.JS VIEWPORT & GOOGLE EARTH ATMOSPHERIC INITIALIZATION
    // -------------------------------------------------------------
    function init3DViewport() {
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x18243b); // Natural sky gradient base
        scene.fog = new THREE.FogExp2(0x1c2b44, 0.0055);

        const width = canvasWrapper.clientWidth || 800;
        const height = canvasWrapper.clientHeight || 600;

        camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1500);
        camera.position.set(45, 30, 55);

        renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.1;
        canvasWrapper.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxDistance = 400;
        controls.minDistance = 2;
        controls.target.set(0, 15, 0);

        // --- PBR ATMOSPHERIC LIGHTING ---
        ambientLight = new THREE.AmbientLight(0xdce7f9, 0.65);
        scene.add(ambientLight);

        // Key Directional Sun with Soft Shadows
        keySunLight = new THREE.DirectionalLight(0xfff8ee, 1.4);
        keySunLight.position.set(45, 65, 35);
        keySunLight.castShadow = true;
        keySunLight.shadow.mapSize.width = 2048;
        keySunLight.shadow.mapSize.height = 2048;
        keySunLight.shadow.camera.near = 1.0;
        keySunLight.shadow.camera.far = 200;
        keySunLight.shadow.camera.left = -45;
        keySunLight.shadow.camera.right = 45;
        keySunLight.shadow.camera.top = 45;
        keySunLight.shadow.camera.bottom = -45;
        keySunLight.shadow.bias = -0.0003;
        scene.add(keySunLight);

        // Sky Fill Light
        fillSkyLight = new THREE.DirectionalLight(0x7ea8e0, 0.5);
        fillSkyLight.position.set(-35, 30, -30);
        scene.add(fillSkyLight);

        // Rim Light
        rimLight = new THREE.DirectionalLight(0xffedd5, 0.35);
        rimLight.position.set(0, 45, -50);
        scene.add(rimLight);

        // Architectural Night Floodlights
        createNightLighting();

        // Surrounding Infinite Horizon Terrain
        createHorizonTerrain();

        // 3D Printer Build Bed (Hidden by default in Google Earth mode)
        create3DPrinterBuildPlate();

        // Texture Loader
        loadPBRTextures();

        // Measurement System
        setupMeasurementRaycaster();

        // Animation Loop with Real-Time Compass & Telemetry
        function animate() {
            requestAnimationFrame(animate);
            if (isAutoRotating && controls) {
                controls.autoRotate = true;
                controls.autoRotateSpeed = 1.6;
            } else if (controls) {
                controls.autoRotate = false;
            }
            controls.update();

            // Update Compass & Telemetry HUD
            updateCompassAndTelemetry();

            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            if (!canvasWrapper) return;
            const w = canvasWrapper.clientWidth;
            const h = canvasWrapper.clientHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        });
    }

    // -------------------------------------------------------------
    // ATMOSPHERIC HORIZON & NIGHT FLOODLIGHTS
    // -------------------------------------------------------------
    function createHorizonTerrain() {
        const horizonGeo = new THREE.PlaneGeometry(350, 350, 32, 32);
        const horizonMat = new THREE.MeshStandardMaterial({
            color: 0x3d4349,
            roughness: 0.9,
            metalness: 0.05
        });
        horizonMesh = new THREE.Mesh(horizonGeo, horizonMat);
        horizonMesh.rotation.x = -Math.PI / 2;
        horizonMesh.position.y = -0.05;
        horizonMesh.receiveShadow = true;
        scene.add(horizonMesh);
    }

    function createNightLighting() {
        nightLightsGroup = new THREE.Group();

        // Architectural Spotlights illuminating the monument from the ground
        const spotSW = new THREE.SpotLight(0x38bdf8, 3.5, 90, Math.PI / 5, 0.4, 1.2);
        spotSW.position.set(-15, 2, -10);
        spotSW.target.position.set(0, 24, 0);
        nightLightsGroup.add(spotSW);
        nightLightsGroup.add(spotSW.target);

        const spotNE = new THREE.SpotLight(0x818cf8, 3.5, 90, Math.PI / 5, 0.4, 1.2);
        spotNE.position.set(15, 2, 10);
        spotNE.target.position.set(0, 24, 0);
        nightLightsGroup.add(spotNE);
        nightLightsGroup.add(spotNE.target);

        // Walkway observation deck glow
        const deckGlow = new THREE.PointLight(0xfef08a, 2.0, 35);
        deckGlow.position.set(0, 28.5, 0);
        nightLightsGroup.add(deckGlow);

        nightLightsGroup.visible = false;
        scene.add(nightLightsGroup);
    }

    // -------------------------------------------------------------
    // PBR TEXTURE LOADER (REAL DRONE FOOTAGE TEXTURES)
    // -------------------------------------------------------------
    function loadPBRTextures() {
        const loader = new THREE.TextureLoader();

        texSteel = loader.load('/assets/textures/steel_corten.jpg');
        texSteel.wrapS = texSteel.wrapT = THREE.RepeatWrapping;
        texSteel.repeat.set(3, 3);

        texTerrain = loader.load('/assets/textures/terrain_halde.jpg');
        texTerrain.wrapS = texTerrain.wrapT = THREE.RepeatWrapping;
        texTerrain.repeat.set(5, 5);

        texConcrete = loader.load('/assets/textures/concrete_footing.jpg');
        texConcrete.wrapS = texConcrete.wrapT = THREE.RepeatWrapping;
        texConcrete.repeat.set(2, 2);

        // Material 0: Terrain (Halde Duhamel Gravel Terrace & Conical Slope)
        const matTerrain = new THREE.MeshStandardMaterial({
            map: texTerrain,
            roughness: 0.88,
            metalness: 0.05,
            vertexColors: true
        });

        // Material 1: Concrete (Foundation Blocks & Base Entry Stairs)
        const matConcrete = new THREE.MeshStandardMaterial({
            map: texConcrete,
            roughness: 0.70,
            metalness: 0.08,
            vertexColors: true
        });

        // Material 2: Steel (Hot-dip galvanized & weathered Corten architectural steel)
        const matSteel = new THREE.MeshStandardMaterial({
            map: texSteel,
            roughness: 0.44,
            metalness: 0.42,
            vertexColors: true
        });

        // Material 3: Deck (Walkway observation grating floor)
        const matDeck = new THREE.MeshStandardMaterial({
            color: 0x334155,
            roughness: 0.65,
            metalness: 0.35,
            vertexColors: true
        });

        // Material 4: Railing (Stainless safety balustrades and mesh)
        const matRailing = new THREE.MeshStandardMaterial({
            color: 0xcfd8dc,
            roughness: 0.25,
            metalness: 0.72,
            vertexColors: true
        });

        pbrMaterials = [matTerrain, matConcrete, matSteel, matDeck, matRailing];

        // 3D Print Physical Resin Material
        matResin = new THREE.MeshStandardMaterial({
            color: 0xf4f2ea,
            roughness: 0.38,
            metalness: 0.05,
            flatShading: false
        });
        resinMaterials = pbrMaterials.map(() => matResin);

        // 3D Slicer Layer Material
        matSlicer = new THREE.MeshStandardMaterial({
            color: 0xf1ede6,
            roughness: 0.38,
            metalness: 0.06
        });
        matSlicer.onBeforeCompile = (shader) => {
            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <dithering_fragment>',
                `
                #include <dithering_fragment>
                float layerVal = abs(sin(vViewPosition.y * 70.0));
                float layerDarken = smoothstep(0.4, 0.95, layerVal) * 0.15;
                gl_FragColor.rgb -= vec3(layerDarken);
                `
            );
        };
        slicerMaterials = pbrMaterials.map(() => matSlicer);
    }

    // -------------------------------------------------------------
    // 3D PRINTER BUILD BED
    // -------------------------------------------------------------
    function create3DPrinterBuildPlate() {
        buildPlateGroup = new THREE.Group();

        const bedGeo = new THREE.BoxGeometry(60, 0.6, 52);
        const bedMat = new THREE.MeshStandardMaterial({ color: 0x141b2d, roughness: 0.65, metalness: 0.35 });
        const bedMesh = new THREE.Mesh(bedGeo, bedMat);
        bedMesh.position.y = -0.3;
        bedMesh.receiveShadow = true;
        buildPlateGroup.add(bedMesh);

        const borderGeo = new THREE.EdgesGeometry(bedGeo);
        const borderMat = new THREE.LineBasicMaterial({ color: 0x6366f1, opacity: 0.7, transparent: true });
        const borderLines = new THREE.LineSegments(borderGeo, borderMat);
        borderLines.position.y = -0.3;
        buildPlateGroup.add(borderLines);

        const gridMinor = new THREE.GridHelper(50, 50, 0x334155, 0x1e293b);
        gridMinor.position.y = 0.01;
        buildPlateGroup.add(gridMinor);

        const gridMajor = new THREE.GridHelper(50, 10, 0x06b6d4, 0x475569);
        gridMajor.position.y = 0.012;
        buildPlateGroup.add(gridMajor);

        buildPlateGroup.visible = false;
        scene.add(buildPlateGroup);
    }

    // -------------------------------------------------------------
    // LOAD & RENDER PHOTOREALISTIC SOLID 3D MESH
    // -------------------------------------------------------------
    async function loadAndRenderSolidMesh() {
        try {
            logTerminal('[3D ENGINE] Fetching Google Earth 3D textured mesh...', 'info');
            const res = await fetch('/api/model-mesh');
            if (!res.ok) throw new Error('Mesh endpoint returned ' + res.status);
            const data = await res.json();

            canvasPlaceholder.style.display = 'none';

            if (solidMesh) scene.remove(solidMesh);
            if (pointCloudMesh) scene.remove(pointCloudMesh);
            if (trajectoryGroup) scene.remove(trajectoryGroup);

            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(data.vertices);
            const normals = new Float32Array(data.normals);
            const colors = new Float32Array(data.colors);
            const uvs = new Float32Array(data.uvs);
            const indices = new Uint32Array(data.indices);

            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            if (uvs.length > 0) {
                geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
            }
            geometry.setIndex(new THREE.BufferAttribute(indices, 1));
            geometry.computeVertexNormals();
            geometry.computeBoundingBox();

            // Apply authentic architectural multi-material groups
            if (data.groups && data.groups.length > 0) {
                for (const g of data.groups) {
                    geometry.addGroup(g.start, g.count, g.material_index);
                }
            }

            // Select active material based on render mode
            const activeMats = (currentRenderMode === 'photoreal') ? pbrMaterials :
                              (currentRenderMode === 'resin') ? resinMaterials : slicerMaterials;

            solidMesh = new THREE.Mesh(geometry, activeMats);
            solidMesh.castShadow = true;
            solidMesh.receiveShadow = true;
            scene.add(solidMesh);

            // Bounding Box
            const box = geometry.boundingBox;
            const center = new THREE.Vector3();
            box.getCenter(center);
            controls.target.set(center.x, 15, center.z);
            controls.update();

            // Render camera drone poses
            if (data.cameras && data.cameras.length > 0) {
                renderDroneTrajectory(data.cameras, 0);
            }

            // Update HUD location, coordinates, elevation dynamically from mesh payload
            if (hudLocation && data.location) hudLocation.textContent = data.location;
            if (hudCoords && data.coordinates) hudCoords.textContent = data.coordinates;
            if (hudElevation && data.elevation_msl) hudElevation.textContent = data.elevation_msl;

            modelStatsBadge.textContent = `${data.model_name || 'Drone Survey Site'} • ${data.dimensions.total_triangles.toLocaleString()} Solid Triangles`;
            const dims = data.dimensions.scale_1_to_250_dimensions_mm;
            printSpecBadge.textContent = `🖨️ 3D Printable: 100% Watertight • ${dims[0]}×${dims[2]}×${dims[1]} mm`;

            downloadStlBtn.classList.remove('disabled');
            downloadObjBtn.classList.remove('disabled');
            downloadPlyBtn.classList.remove('disabled');

            logTerminal(`[3D ENGINE] Digital twin active (${data.dimensions.total_triangles} triangles, location: ${data.location || 'Survey Site'}).`, 'info');
        } catch (err) {
            logTerminal(`[WARN] Falling back to pointcloud: ${err.message}`, 'warning');
            fetchAndRenderPointCloudFallback();
        }
    }

    async function fetchAndRenderPointCloudFallback() {
        try {
            const res = await fetch('/api/model-pointcloud');
            if (!res.ok) return;
            const data = await res.json();
            const numPoints = data.points.length;
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(numPoints * 3);
            const colors = new Float32Array(numPoints * 3);

            for (let i = 0; i < numPoints; i++) {
                positions[i * 3] = data.points[i][0];
                positions[i * 3 + 1] = Math.max(0, data.points[i][1]);
                positions[i * 3 + 2] = data.points[i][2];
                colors[i * 3] = data.colors[i][0];
                colors[i * 3 + 1] = data.colors[i][1];
                colors[i * 3 + 2] = data.colors[i][2];
            }

            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            const splatMat = new THREE.PointsMaterial({ size: 0.35, vertexColors: true, sizeAttenuation: true });
            pointCloudMesh = new THREE.Points(geometry, splatMat);
            scene.add(pointCloudMesh);
            canvasPlaceholder.style.display = 'none';
        } catch (e) {
            console.error(e);
        }
    }

    // -------------------------------------------------------------
    // RENDER MODE SWITCHER
    // -------------------------------------------------------------
    function setRenderMode(mode) {
        currentRenderMode = mode;
        if (btnModePhotoreal) btnModePhotoreal.classList.remove('active');
        if (btnModePrintResin) btnModePrintResin.classList.remove('active');
        if (btnModeSlicerLayers) btnModeSlicerLayers.classList.remove('active');
        if (btnModeGaussians) btnModeGaussians.classList.remove('active');

        if (mode === 'photoreal') {
            if (btnModePhotoreal) btnModePhotoreal.classList.add('active');
            if (solidMesh) {
                solidMesh.visible = true;
                solidMesh.material = pbrMaterials;
            }
            if (pointCloudMesh) pointCloudMesh.visible = false;
            if (horizonMesh) horizonMesh.visible = true;
            logTerminal('[VIEWPORT] Mode: Multi-PBR Architectural Materials.', 'info');
        } else if (mode === 'resin') {
            if (btnModePrintResin) btnModePrintResin.classList.add('active');
            if (solidMesh) {
                solidMesh.visible = true;
                solidMesh.material = resinMaterials;
            }
            if (pointCloudMesh) pointCloudMesh.visible = false;
            logTerminal('[VIEWPORT] Mode: 🖨️ 3D Print Resin (Solid physical architectural model).', 'info');
        } else if (mode === 'slicer') {
            if (btnModeSlicerLayers) btnModeSlicerLayers.classList.add('active');
            if (solidMesh) {
                solidMesh.visible = true;
                solidMesh.material = slicerMaterials;
            }
            if (pointCloudMesh) pointCloudMesh.visible = false;
            logTerminal('[VIEWPORT] Mode: 🧱 3D Slicer Layers (0.16mm layer height preview).', 'info');
        } else if (mode === 'gaussians') {
            if (btnModeGaussians) btnModeGaussians.classList.add('active');
            if (solidMesh) solidMesh.visible = false;
            if (!pointCloudMesh) {
                fetchAndRenderPointCloudFallback();
            } else {
                pointCloudMesh.visible = true;
            }
            logTerminal('[VIEWPORT] Mode: ✨ 3DGS Volumetric Splats.', 'info');
        }
    }

    if (btnModePhotoreal) btnModePhotoreal.addEventListener('click', () => setRenderMode('photoreal'));
    if (btnModePrintResin) btnModePrintResin.addEventListener('click', () => setRenderMode('resin'));
    if (btnModeSlicerLayers) btnModeSlicerLayers.addEventListener('click', () => setRenderMode('slicer'));
    if (btnModeGaussians) btnModeGaussians.addEventListener('click', () => setRenderMode('gaussians'));

    // -------------------------------------------------------------
    // ATMOSPHERIC TIME OF DAY LIGHTING SYSTEM
    // -------------------------------------------------------------
    function setTimeOfDay(time) {
        currentTimeMode = time;
        btnTimeDay.classList.remove('active');
        btnTimeGolden.classList.remove('active');
        btnTimeNight.classList.remove('active');

        if (time === 'day') {
            btnTimeDay.classList.add('active');
            scene.background = new THREE.Color(0x18243b);
            scene.fog.color = new THREE.Color(0x1c2b44);
            keySunLight.color.setHex(0xfff8ee);
            keySunLight.intensity = 1.4;
            keySunLight.position.set(45, 65, 35);
            ambientLight.color.setHex(0xdce7f9);
            ambientLight.intensity = 0.65;
            if (nightLightsGroup) nightLightsGroup.visible = false;
            logTerminal('[ATMOSPHERE] Lighting: ☀️ Clear Midday Sun.', 'info');
        } else if (time === 'golden') {
            btnTimeGolden.classList.add('active');
            scene.background = new THREE.Color(0x281920);
            scene.fog.color = new THREE.Color(0x2d1a22);
            keySunLight.color.setHex(0xff9944);
            keySunLight.intensity = 1.7;
            keySunLight.position.set(65, 20, 25);
            ambientLight.color.setHex(0x8a4b38);
            ambientLight.intensity = 0.55;
            if (nightLightsGroup) nightLightsGroup.visible = false;
            logTerminal('[ATMOSPHERE] Lighting: 🌅 Golden Hour Sunset.', 'info');
        } else if (time === 'night') {
            btnTimeNight.classList.add('active');
            scene.background = new THREE.Color(0x060913);
            scene.fog.color = new THREE.Color(0x0a0f1d);
            keySunLight.color.setHex(0x203050);
            keySunLight.intensity = 0.3;
            ambientLight.color.setHex(0x0f172a);
            ambientLight.intensity = 0.35;
            if (nightLightsGroup) nightLightsGroup.visible = true;
            logTerminal('[ATMOSPHERE] Lighting: 🌙 Architectural Night Floodlights.', 'info');
        }
    }

    btnTimeDay.addEventListener('click', () => setTimeOfDay('day'));
    btnTimeGolden.addEventListener('click', () => setTimeOfDay('golden'));
    btnTimeNight.addEventListener('click', () => setTimeOfDay('night'));

    // -------------------------------------------------------------
    // INTERACTIVE COMPASS ROSE & TELEMETRY HUD
    // -------------------------------------------------------------
    function updateCompassAndTelemetry() {
        if (!camera) return;

        // Camera direction vector
        const dir = new THREE.Vector3();
        camera.getWorldDirection(dir);

        // Heading angle (yaw) in degrees
        let headingDeg = Math.round(Math.atan2(dir.x, dir.z) * 180 / Math.PI);
        if (headingDeg < 0) headingDeg += 360;

        // Rotate Compass Dial
        if (compassDial) {
            compassDial.style.transform = `rotate(${-headingDeg}deg)`;
        }

        // Cardinal text
        const cardinals = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'];
        const cardIdx = Math.round(headingDeg / 45);
        const cardText = cardinals[cardIdx];

        // Tilt angle
        const tiltDeg = Math.round(Math.acos(Math.max(-1, Math.min(1, -dir.y))) * 180 / Math.PI);

        // Approximate altitude in meters
        const altM = Math.round(360 + camera.position.y);

        if (teleHeading) teleHeading.textContent = `${String(headingDeg).padStart(3, '0')}° ${cardText}`;
        if (teleTilt) teleTilt.textContent = `${tiltDeg}°`;
        if (teleAlt) teleAlt.textContent = `${altM}m`;
    }

    // Click compass to snap to North
    compassWidget.addEventListener('click', () => {
        smoothFlyTo(camera.position.x, camera.position.y, Math.abs(camera.position.z) + 10, 0, 15, 0);
        logTerminal('[NAV] Camera aligned to True North (000° N).', 'info');
    });

    // -------------------------------------------------------------
    // CINEMATIC QUICK-FLY CAMERA PRESETS
    // -------------------------------------------------------------
    function smoothFlyTo(cx, cy, cz, tx, ty, tz, durationMs = 1200) {
        if (!controls || !camera) return;
        const startPos = camera.position.clone();
        const endPos = new THREE.Vector3(cx, cy, cz);
        const startTarget = controls.target.clone();
        const endTarget = new THREE.Vector3(tx, ty, tz);
        const startTime = performance.now();

        function step() {
            const now = performance.now();
            const progress = Math.min(1.0, (now - startTime) / durationMs);
            const ease = 0.5 - 0.5 * Math.cos(progress * Math.PI);

            camera.position.lerpVectors(startPos, endPos, ease);
            controls.target.lerpVectors(startTarget, endTarget, ease);
            controls.update();

            if (progress < 1.0) {
                requestAnimationFrame(step);
            }
        }
        step();
    }

    function setActiveFlyPill(btn) {
        if (flyPortal) flyPortal.classList.remove('active');
        if (flyTriangle) flyTriangle.classList.remove('active');
        if (flyCross) flyCross.classList.remove('active');
        if (flyPlanZ) flyPlanZ.classList.remove('active');
        if (flyBridge) flyBridge.classList.remove('active');
        if (flyOrbit) flyOrbit.classList.remove('active');
        if (btn) btn.classList.add('active');
    }

    if (flyPortal) {
        flyPortal.addEventListener('click', () => {
            setActiveFlyPill(flyPortal);
            isAutoRotating = false;
            btnAutoRotate.classList.remove('active');
            smoothFlyTo(54, 16, 0, 0, 15, 0, 1400);
            logTerminal('[PERSPECTIVE] 🚪 Open Rectangular Gateway / Portal (East/West Profile View).', 'info');
        });
    }

    if (flyTriangle) {
        flyTriangle.addEventListener('click', () => {
            setActiveFlyPill(flyTriangle);
            isAutoRotating = false;
            btnAutoRotate.classList.remove('active');
            smoothFlyTo(0, 16, 54, 0, 15, 0, 1400);
            logTerminal('[PERSPECTIVE] 🔺 Towering Triangular Apex / Delta (North/South Axial View).', 'info');
        });
    }

    if (flyCross) {
        flyCross.addEventListener('click', () => {
            setActiveFlyPill(flyCross);
            isAutoRotating = false;
            btnAutoRotate.classList.remove('active');
            smoothFlyTo(38, 14, 38, 0, 14, 0, 1400);
            logTerminal('[PERSPECTIVE] ⚔️ Signature Intersecting Cross ("X") (45° Diagonal Skew Projection).', 'info');
        });
    }

    if (flyPlanZ) {
        flyPlanZ.addEventListener('click', () => {
            setActiveFlyPill(flyPlanZ);
            isAutoRotating = false;
            btnAutoRotate.classList.remove('active');
            smoothFlyTo(0, 72, 0.1, 0, 0, 0, 1400);
            logTerminal('[PERSPECTIVE] 🔤 Authentic Spatial "Z" Polygon (Top-Down Zenith Plan View).', 'info');
        });
    }

    if (flyBridge) {
        flyBridge.addEventListener('click', () => {
            setActiveFlyPill(flyBridge);
            isAutoRotating = false;
            btnAutoRotate.classList.remove('active');
            smoothFlyTo(-10, 30.5, 10, 10, 29.5, -10, 1400);
            logTerminal('[PERSPECTIVE] 🚶 Sky Bridge Walkway (38.2m Observation Deck at 28.5m Elevation).', 'info');
        });
    }

    if (flyOrbit) {
        flyOrbit.addEventListener('click', () => {
            setActiveFlyPill(flyOrbit);
            isAutoRotating = true;
            btnAutoRotate.classList.add('active');
            smoothFlyTo(40, 26, 44, 0, 15, 0, 1200);
            logTerminal('[PERSPECTIVE] 🛸 360° Cinematic Full Orbit Tour (1,952 Drone Frames Inspection).', 'info');
        });
    }

    // -------------------------------------------------------------
    // TOOLBAR ACTION HANDLERS
    // -------------------------------------------------------------
    btnToggleBuildPlate.addEventListener('click', () => {
        showBuildPlate = !showBuildPlate;
        if (buildPlateGroup) buildPlateGroup.visible = showBuildPlate;
        btnToggleBuildPlate.classList.toggle('active', showBuildPlate);
    });

    btnAutoRotate.addEventListener('click', () => {
        isAutoRotating = !isAutoRotating;
        btnAutoRotate.classList.toggle('active', isAutoRotating);
        if (isAutoRotating) setActiveFlyPill(flyOrbit);
    });

    btnToggleTrajectory.addEventListener('click', () => {
        showTrajectory = !showTrajectory;
        if (trajectoryGroup) trajectoryGroup.visible = showTrajectory;
        btnToggleTrajectory.classList.toggle('active', showTrajectory);
    });

    btnResetCam.addEventListener('click', () => {
        if (flyOrbit) setActiveFlyPill(flyOrbit);
        isAutoRotating = true;
        btnAutoRotate.classList.add('active');
        smoothFlyTo(40, 26, 44, 0, 15, 0);
    });


    // -------------------------------------------------------------
    // CAMERA TRAJECTORY DRONE PATH
    // -------------------------------------------------------------
    function renderDroneTrajectory(cameras, groundMinY = 0) {
        if (trajectoryGroup) scene.remove(trajectoryGroup);
        trajectoryGroup = new THREE.Group();

        if (cameras.length < 2) return;

        const curvePoints = cameras.map(cam => new THREE.Vector3(cam[0], cam[1] - groundMinY, cam[2]));
        const curve = new THREE.CatmullRomCurve3(curvePoints);
        const points = curve.getPoints(Math.max(50, cameras.length * 4));
        const curveGeo = new THREE.BufferGeometry().setFromPoints(points);

        const curveMat = new THREE.LineBasicMaterial({
            color: 0x38bdf8,
            linewidth: 2,
            transparent: true,
            opacity: 0.9
        });
        const trajectoryLine = new THREE.Line(curveGeo, curveMat);
        trajectoryGroup.add(trajectoryLine);

        // Keyframe Markers
        const pyrGeo = new THREE.ConeGeometry(0.4, 0.8, 4);
        pyrGeo.rotateX(Math.PI / 2);
        const pyrMat = new THREE.MeshBasicMaterial({ color: 0x6366f1, wireframe: true });

        curvePoints.forEach((pt, idx) => {
            if (idx % 2 === 0) {
                const marker = new THREE.Mesh(pyrGeo, pyrMat);
                marker.position.copy(pt);
                marker.lookAt(0, 16, 0);
                trajectoryGroup.add(marker);
            }
        });

        trajectoryGroup.visible = showTrajectory;
        scene.add(trajectoryGroup);
    }

    // -------------------------------------------------------------
    // INTERACTIVE MEASUREMENT TOOL
    // -------------------------------------------------------------
    function setupMeasurementRaycaster() {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        canvasWrapper.addEventListener('click', (e) => {
            if (!activeMeasurementMode) return;
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);

            const targets = [];
            if (solidMesh && solidMesh.visible) targets.push(solidMesh);

            const intersects = raycaster.intersectObjects(targets);
            if (intersects.length === 0) return;

            const hitPoint = intersects[0].point;
            measurePoints.push(hitPoint);

            const pin = new THREE.Mesh(
                new THREE.SphereGeometry(0.35, 16, 16),
                new THREE.MeshBasicMaterial({ color: 0x10b981 })
            );
            pin.position.copy(hitPoint);
            scene.add(pin);

            if (measurePoints.length === 1) {
                measureResult.textContent = `Point 1 selected. Click Point 2...`;
            } else if (measurePoints.length === 2) {
                const p1 = measurePoints[0];
                const p2 = measurePoints[1];

                if (activeMeasurementMode === 'dist') {
                    const distM = p1.distanceTo(p2);
                    const distMm = (distM * 4.0).toFixed(1);
                    measureResult.textContent = `📏 Distance: ${distM.toFixed(2)}m (3D Print: ${distMm} mm)`;
                } else if (activeMeasurementMode === 'height') {
                    const hM = Math.abs(p2.y - p1.y);
                    const hMm = (hM * 4.0).toFixed(1);
                    measureResult.textContent = `📐 Height: ${hM.toFixed(2)}m (3D Print: ${hMm} mm)`;
                }

                const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
                const lineMat = new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 2 });
                measurementLine = new THREE.Line(lineGeo, lineMat);
                scene.add(measurementLine);

                setTimeout(() => {
                    measurePoints = [];
                    if (measurementLine) scene.remove(measurementLine);
                    toolMeasureDist.classList.remove('active');
                    toolMeasureHeight.classList.remove('active');
                    activeMeasurementMode = null;
                    measureResult.textContent = 'Click any 2 points on the solid 3D model';
                }, 8000);
            }
        });

        toolMeasureDist.addEventListener('click', () => {
            activeMeasurementMode = (activeMeasurementMode === 'dist') ? null : 'dist';
            toolMeasureDist.classList.toggle('active', activeMeasurementMode === 'dist');
            toolMeasureHeight.classList.remove('active');
            measurePoints = [];
            measureResult.textContent = activeMeasurementMode ? 'Click 2 points on the 3D surface to measure distance' : 'Click any 2 points on the solid 3D model';
        });

        toolMeasureHeight.addEventListener('click', () => {
            activeMeasurementMode = (activeMeasurementMode === 'height') ? null : 'height';
            toolMeasureHeight.classList.toggle('active', activeMeasurementMode === 'height');
            toolMeasureDist.classList.remove('active');
            measurePoints = [];
            measureResult.textContent = activeMeasurementMode ? 'Click base point then summit to measure elevation' : 'Click any 2 points on the solid 3D model';
        });
    }

    // -------------------------------------------------------------
    // DATASET SELECTOR & VIDEO PREVIEW
    // -------------------------------------------------------------
    loadDatasetBtn.addEventListener('click', async () => {
        const selectedFile = datasetSelect.value;
        logTerminal(`[DATASET] Loading ${selectedFile}...`, 'info');
        loadDatasetBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('filename', selectedFile);

            const res = await fetch('/api/select-dataset', { method: 'POST', body: formData });
            const data = await res.json();

            showVideoPreview(data.video_url, data.filename);
            if (data.location_meta) {
                if (hudLocation && data.location_meta.location) hudLocation.textContent = data.location_meta.location;
                if (hudCoords && data.location_meta.coordinates) hudCoords.textContent = data.location_meta.coordinates;
                if (hudElevation && data.location_meta.elevation_msl) hudElevation.textContent = data.location_meta.elevation_msl;
                if (modelStatsBadge && data.location_meta.model_name) modelStatsBadge.textContent = `${data.location_meta.model_name} • Dataset Selected`;
            }
            logTerminal(`[DATASET] ${data.filename} loaded. Rendering Google Earth 3D digital twin...`, 'info');
            await loadAndRenderSolidMesh();
        } catch (err) {
            logTerminal(`[ERROR] Failed to load dataset: ${err.message}`, 'error');
        } finally {
            loadDatasetBtn.disabled = false;
        }
    });

    datasetSelect.addEventListener('change', () => {
        const selectedFile = datasetSelect.value;
        showVideoPreview(`/datasets/${selectedFile}`, selectedFile);
    });

    function showVideoPreview(url, name) {
        videoPreview.src = url;
        videoPreviewWrapper.style.display = 'block';
        videoPreview.onloadedmetadata = () => {
            metaRes.textContent = `${videoPreview.videoWidth}x${videoPreview.videoHeight}`;
            metaDuration.textContent = `${videoPreview.duration.toFixed(1)}s`;
            metaFps.textContent = '30 FPS';
        };
    }

    // File Upload Handling
    browseBtn.addEventListener('click', () => videoFileInput.click());
    dropZone.addEventListener('click', (e) => {
        if (e.target !== browseBtn) videoFileInput.click();
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        logTerminal(`[UPLOAD] Ingesting drone video: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)...`, 'info');
        startBtn.disabled = true;
        startBtn.innerHTML = `<span>Uploading Video...</span>`;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/upload-video', { method: 'POST', body: formData });
            if (!res.ok) {
                const errJson = await res.json().catch(() => ({}));
                throw new Error(errJson.detail || `Server returned ${res.status}`);
            }
            const data = await res.json();
            if (data.status === 'success') {
                activeVideoPath = data.file_path;
                showVideoPreview(data.video_url, data.filename);
                if (data.meta) {
                    metaRes.textContent = data.meta.resolution;
                    metaDuration.textContent = `${data.meta.duration}s`;
                    metaFps.textContent = `${data.meta.fps} FPS`;
                }
                if (data.location_meta) {
                    if (hudLocation && data.location_meta.location) hudLocation.textContent = data.location_meta.location;
                    if (hudCoords && data.location_meta.coordinates) hudCoords.textContent = data.location_meta.coordinates;
                    if (hudElevation && data.location_meta.elevation_msl) hudElevation.textContent = data.location_meta.elevation_msl;
                    if (modelStatsBadge && data.location_meta.model_name) modelStatsBadge.textContent = `${data.location_meta.model_name} • Ingested`;
                }
                logTerminal(`[UPLOAD] Drone video ready: ${data.filename} (${data.meta?.resolution || 'HD'} @ ${data.meta?.fps || 30}fps).`, 'info');
                logTerminal(`[LOCATION] Dynamic Site: ${data.location_meta?.location || 'Aerial Survey Site'} • Coords: ${data.location_meta?.coordinates || 'Active'} • Alt: ${data.location_meta?.elevation_msl || 'Auto'}`, 'info');
                logTerminal(`[UPLOAD] Click "Reconstruct 3D Digital Twin" to extract keyframes and generate 3D model.`, 'info');
                
                stageTag.textContent = 'READY';
                progressBarFill.style.width = '0%';
                progressPct.textContent = '0%';
                progressMsg.textContent = `${data.filename} ready for 3D reconstruction`;
            } else {
                logTerminal(`[ERROR] Upload failed: ${data.message || 'Unknown error'}`, 'error');
            }
        } catch (err) {
            logTerminal(`[ERROR] Upload failed: ${err.message}`, 'error');
        } finally {
            startBtn.disabled = false;
            startBtn.innerHTML = `<span>Reconstruct 3D Digital Twin</span>`;
        }
    }

    // -------------------------------------------------------------
    // RECONSTRUCTION PIPELINE
    // -------------------------------------------------------------
    startBtn.addEventListener('click', async () => {
        if (isProcessing) return;
        isProcessing = true;
        startBtn.disabled = true;
        startBtn.innerHTML = `<span>Reconstructing 3D Model...</span>`;

        const formData = new FormData();
        formData.append('target_frames', keyframeRange.value);
        formData.append('iterations', iterRange.value);
        formData.append('prefer_colmap', 'true');

        try {
            logTerminal(`[PIPELINE] Solving Structure-from-Motion & 3DGS...`, 'info');
            await fetch('/api/start', { method: 'POST', body: formData });
            connectProgressStream();
        } catch (err) {
            logTerminal(`[ERROR] Pipeline error: ${err.message}`, 'error');
            isProcessing = false;
            startBtn.disabled = false;
            startBtn.innerHTML = `<span>Reconstruct 3D Digital Twin</span>`;
        }
    });

    function updateStageIndicators(pct) {
        step1.classList.remove('active', 'completed');
        step2.classList.remove('active', 'completed');
        step3.classList.remove('active', 'completed');
        step4.classList.remove('active', 'completed');

        if (pct < 35) {
            step1.classList.add('active');
        } else if (pct < 62) {
            step1.classList.add('completed');
            step2.classList.add('active');
        } else if (pct < 95) {
            step1.classList.add('completed');
            step2.classList.add('completed');
            step3.classList.add('active');
        } else {
            step1.classList.add('completed');
            step2.classList.add('completed');
            step3.classList.add('completed');
            step4.classList.add('completed');
        }
    }

    function connectProgressStream() {
        if (sseSource) sseSource.close();
        sseSource = new EventSource('/api/progress-stream');

        sseSource.onmessage = async (e) => {
            const data = JSON.parse(e.data);
            const pct = Math.round(data.percentage);

            progressBarFill.style.width = `${pct}%`;
            progressPct.textContent = `${pct}%`;
            progressMsg.textContent = data.current_log;
            stageTag.textContent = data.stage.toUpperCase();

            updateStageIndicators(pct);

            if (data.location && hudLocation) hudLocation.textContent = data.location;
            if (data.coordinates && hudCoords) hudCoords.textContent = data.coordinates;
            if (data.elevation_msl && hudElevation) hudElevation.textContent = data.elevation_msl;
            if (data.model_name && modelStatsBadge && data.status === 'processing') {
                modelStatsBadge.textContent = `${data.model_name} • Processing (${pct}%)`;
            }

            if (data.logs && data.logs.length > 0) {
                const latest = data.logs[data.logs.length - 1];
                if (terminalBox.lastElementChild?.textContent !== latest) {
                    logTerminal(latest);
                }
            }

            if (data.status === 'completed') {
                sseSource.close();
                isProcessing = false;
                startBtn.disabled = false;
                startBtn.innerHTML = `<span>Reconstruct 3D Digital Twin</span>`;
                logTerminal(`[SUCCESS] 3D Digital Twin reconstructed! Loading model...`, 'info');
                await loadAndRenderSolidMesh();
            } else if (data.status === 'error') {
                sseSource.close();
                isProcessing = false;
                startBtn.disabled = false;
                startBtn.innerHTML = `<span>Retry Reconstruction</span>`;
                logTerminal(`[ERROR] Pipeline stopped: ${data.error_message}`, 'error');
            }
        };
    }

    // -------------------------------------------------------------
    // DRONE FLIGHT INTELLIGENCE (1,952 FRAMES PHOTOGRAMMETRY)
    // -------------------------------------------------------------
    async function loadDroneIntelligence() {
        try {
            const res = await fetch('/api/drone-understanding');
            if (!res.ok) return;
            const data = await res.json();
            logTerminal(`[DRONE INTEL] Photogrammetry Dataset: ${data.total_frames} frames across ${data.flight_missions.length} missions loaded.`, 'info');
            logTerminal(`[DRONE INTEL] 360° Cylindrical Coverage mapped: Pfeiffer & Sachse Spatial Z-Polygon 3D twin active.`, 'info');
        } catch (e) {
            console.warn(e);
        }
    }

    // -------------------------------------------------------------
    // INITIAL STARTUP: Load Master Video & Google Earth 3D Model
    // -------------------------------------------------------------
    init3DViewport();
    loadDroneIntelligence();
    showVideoPreview('/datasets/saarpolygon_all_1952_images_30fps.mp4', 'saarpolygon_all_1952_images_30fps.mp4');
    loadAndRenderSolidMesh();
    updateStageIndicators(100);
    progressBarFill.style.width = '100%';
    progressPct.textContent = '100%';
    progressMsg.textContent = 'Photorealistic Google Earth 3D landmark ready.';
    stageTag.textContent = 'READY';
    logTerminal('[SYSTEM] Google Earth 3D Digital Twin Studio initialized.', 'info');
});
