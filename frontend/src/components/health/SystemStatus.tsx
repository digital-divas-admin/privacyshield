"use client";

import { useHealth } from "@/hooks/useHealth";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { API_URL } from "@/lib/constants";

function StatusRow({ label, loaded, detail }: { label: string; loaded: boolean; detail?: string }) {
  return (
    <div className="flex items-center justify-between py-2">
      <div>
        <span className="text-sm font-medium">{label}</span>
        {detail && <span className="ml-2 text-xs text-muted-foreground">{detail}</span>}
      </div>
      <Badge variant={loaded ? "success" : "outline"}>
        {loaded ? "Loaded" : "Not Available"}
      </Badge>
    </div>
  );
}

export function SystemStatus() {
  const { health, connected, error, refresh } = useHealth();

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      {/* Connection Status */}
      <Card>
        <CardHeader>
          <CardTitle>API Connection</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">{API_URL}</p>
              {error && <p className="mt-1 text-xs text-red-400">{error}</p>}
            </div>
            <div className="flex items-center gap-3">
              <Badge variant={connected ? "success" : "destructive"}>
                {connected ? "Connected" : "Offline"}
              </Badge>
              <Button variant="outline" size="sm" onClick={refresh}>
                Refresh
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Device Info */}
      {health && (
        <Card>
          <CardHeader>
            <CardTitle>Device</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  "inline-block rounded px-2 py-1 text-xs font-mono font-bold",
                  health.device === "cuda" ? "bg-green-600/20 text-green-400" : "bg-yellow-600/20 text-yellow-400"
                )}
              >
                {health.device.toUpperCase()}
              </span>
              {health.device === "cpu" && (
                <span className="text-xs text-yellow-400">GPU recommended for real-time processing</span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Status */}
      {health && (
        <Card>
          <CardHeader>
            <CardTitle>Models</CardTitle>
          </CardHeader>
          <CardContent className="divide-y divide-border">
            <StatusRow
              label="ArcFace (IResNet-100)"
              loaded={health.face_model_loaded}
              detail="Face recognition model — required for all modes"
            />
            <StatusRow
              label="U-Net Encoder"
              loaded={health.encoder_loaded}
              detail="Single-pass encoder — requires training"
            />
            <StatusRow
              label="ViT-S/8 Encoder"
              loaded={health.vit_encoder_loaded}
              detail="Production encoder — requires training"
            />
            <StatusRow
              label="V2 Pipeline"
              loaded={health.pipeline_v2_loaded}
              detail="LPIPS + CLIP + semantic mask"
            />
            <StatusRow
              label="FaceNet (InceptionResNet-V1)"
              loaded={health.facenet_loaded}
              detail="Ensemble member — VGGFace2 pretrained"
            />
            <StatusRow
              label="AdaFace (IR-101)"
              loaded={health.adaface_loaded}
              detail="Ensemble member — requires weights download"
            />
            <StatusRow
              label="Inswapper (Roop)"
              loaded={health.inswapper_loaded}
              detail="Face swap model — lazy-loaded on first deepfake test"
            />
            <StatusRow
              label="IP-Adapter FaceID Plus v2"
              loaded={health.ipadapter_loaded}
              detail="Generative deepfake — requires diffusers"
            />
          </CardContent>
        </Card>
      )}

      {/* Ensemble Status */}
      {health && health.ensemble_models.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle>Ensemble Attack</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-2 text-xs text-muted-foreground">
              Cross-model ensemble optimizes perturbations against multiple FR models for improved transferability.
            </p>
            <div className="flex flex-wrap gap-2">
              {health.ensemble_models.map((name) => (
                <Badge key={name} variant="success">{name}</Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Available Modes */}
      {health && (
        <Card>
          <CardHeader>
            <CardTitle>Available Modes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {health.face_model_loaded && (
                <Badge variant="success">pgd</Badge>
              )}
              {health.pipeline_v2_loaded && (
                <>
                  <Badge variant="success">v2</Badge>
                  <Badge variant="success">v2_full</Badge>
                </>
              )}
              {health.encoder_loaded && (
                <Badge variant="success">encoder</Badge>
              )}
              {health.vit_encoder_loaded && (
                <Badge variant="success">vit</Badge>
              )}
              {!health.face_model_loaded && (
                <span className="text-xs text-red-400">No modes available — face model not loaded</span>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
