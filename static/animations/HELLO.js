window.ISL_ANIM_HELLO = (ref) => {

    // Frame 1: Raise right arm up and out for waving
    let animations = [];
    animations.push(["mixamorigRightArm", "rotation", "z", -Math.PI/2.5, "-"]);
    animations.push(["mixamorigRightForeArm", "rotation", "x", -Math.PI/2, "-"]);
    animations.push(["mixamorigRightHand", "rotation", "x", 0, "+"]);
    // Open hand
    animations.push(["mixamorigRightHandThumb2", "rotation", "y", -Math.PI/6, "-"]);
    ref.animations.push(animations);

    // Frame 2: Wave left (inward)
    animations = [];
    animations.push(["mixamorigRightForeArm", "rotation", "z", Math.PI/4, "+"]);
    ref.animations.push(animations);

    // Frame 3: Wave right (outward)
    animations = [];
    animations.push(["mixamorigRightForeArm", "rotation", "z", -Math.PI/8, "-"]);
    ref.animations.push(animations);
    
    // Frame 4: Wave left (inward) again
    animations = [];
    animations.push(["mixamorigRightForeArm", "rotation", "z", Math.PI/4, "+"]);
    ref.animations.push(animations);

    // Frame 5: Return to neutral default pose (arms dropped, not T-Pose)
    animations = [];
    animations.push(["mixamorigRightArm", "rotation", "z", Math.PI/3, "+"]);
    animations.push(["mixamorigRightForeArm", "rotation", "x", 0, "+"]);
    animations.push(["mixamorigRightForeArm", "rotation", "y", Math.PI/1.5, "+"]);
    animations.push(["mixamorigRightForeArm", "rotation", "z", 0, "-"]);
    animations.push(["mixamorigRightHandThumb2", "rotation", "y", 0, "+"]);
    ref.animations.push(animations);

    if(ref.pending === false){
        ref.pending = true;
        ref.animate();
    }

}
