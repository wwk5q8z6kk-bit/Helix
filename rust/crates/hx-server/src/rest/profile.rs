use std::sync::Arc;

use axum::{
	extract::State,
	http::StatusCode,
	response::IntoResponse,
	Extension, Json,
};
use serde::{Deserialize, Serialize};

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Response / Request DTOs
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ProfileResponse {
	pub display_name: String,
	pub avatar_url: Option<String>,
	pub bio: Option<String>,
	pub email: Option<String>,
	pub preferred_namespace: String,
	pub default_node_kind: String,
	pub preferred_llm_provider: Option<String>,
	pub timezone: String,
	pub signature_name: Option<String>,
	pub signature_public_key: Option<String>,
	pub metadata: std::collections::HashMap<String, serde_json::Value>,
	pub created_at: String,
	pub updated_at: String,
}

#[derive(Deserialize)]
pub struct UpdateProfileRequest {
	pub display_name: Option<String>,
	pub avatar_url: Option<String>,
	pub bio: Option<String>,
	pub email: Option<String>,
	pub preferred_namespace: Option<String>,
	pub default_node_kind: Option<String>,
	pub preferred_llm_provider: Option<String>,
	pub timezone: Option<String>,
	pub signature_name: Option<String>,
	pub signature_public_key: Option<String>,
	pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
}

#[derive(Serialize)]
struct ErrorBody {
	error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /api/v1/profile
pub async fn get_profile(
	Extension(auth): Extension<AuthContext>,
	State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
	if let Err(err) = authorize_read(&auth) {
		return (
			err.0,
			Json(ErrorBody {
				error: err.1,
			}),
		)
			.into_response();
	}

	match state.engine.get_profile().await {
		Ok(p) => Json(ProfileResponse {
			display_name: p.display_name,
			avatar_url: p.avatar_url,
			bio: p.bio,
			email: p.email,
			preferred_namespace: p.preferred_namespace,
			default_node_kind: p.default_node_kind,
			preferred_llm_provider: p.preferred_llm_provider,
			timezone: p.timezone,
			signature_name: p.signature_name,
			signature_public_key: p.signature_public_key,
			metadata: p.metadata,
			created_at: p.created_at.to_rfc3339(),
			updated_at: p.updated_at.to_rfc3339(),
		})
		.into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(ErrorBody {
				error: format!("failed to read profile: {e}"),
			}),
		)
			.into_response(),
	}
}

/// PUT /api/v1/profile
pub async fn update_profile(
	Extension(auth): Extension<AuthContext>,
	State(state): State<Arc<AppState>>,
	Json(req): Json<UpdateProfileRequest>,
) -> impl IntoResponse {
	if let Err(err) = authorize_write(&auth) {
		return (
			err.0,
			Json(ErrorBody {
				error: err.1,
			}),
		)
			.into_response();
	}

	// Validate display_name not empty when provided
	if let Some(ref name) = req.display_name {
		if name.trim().is_empty() {
			return (
				StatusCode::BAD_REQUEST,
				Json(ErrorBody {
					error: "display_name cannot be empty".into(),
				}),
			)
				.into_response();
		}
	}

	// Validate timezone format (IANA-style: must contain '/' or be "UTC")
	if let Some(ref tz) = req.timezone {
		let tz_trimmed = tz.trim();
		if tz_trimmed.is_empty()
			|| (!tz_trimmed.eq_ignore_ascii_case("UTC")
				&& !tz_trimmed.contains('/'))
		{
			return (
				StatusCode::BAD_REQUEST,
				Json(ErrorBody {
					error: "timezone must be 'UTC' or IANA format (e.g. 'America/New_York')".into(),
				}),
			)
				.into_response();
		}
	}

	let core_req = hx_core::UpdateProfileRequest {
		display_name: req.display_name,
		avatar_url: req.avatar_url,
		bio: req.bio,
		email: req.email,
		preferred_namespace: req.preferred_namespace,
		default_node_kind: req.default_node_kind,
		preferred_llm_provider: req.preferred_llm_provider,
		timezone: req.timezone,
		signature_name: req.signature_name,
		signature_public_key: req.signature_public_key,
		metadata: req.metadata,
	};

	match state.engine.update_profile(&core_req).await {
		Ok(p) => Json(ProfileResponse {
			display_name: p.display_name,
			avatar_url: p.avatar_url,
			bio: p.bio,
			email: p.email,
			preferred_namespace: p.preferred_namespace,
			default_node_kind: p.default_node_kind,
			preferred_llm_provider: p.preferred_llm_provider,
			timezone: p.timezone,
			signature_name: p.signature_name,
			signature_public_key: p.signature_public_key,
			metadata: p.metadata,
			created_at: p.created_at.to_rfc3339(),
			updated_at: p.updated_at.to_rfc3339(),
		})
		.into_response(),
		Err(e) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Json(ErrorBody {
				error: format!("failed to update profile: {e}"),
			}),
		)
			.into_response(),
	}
}
