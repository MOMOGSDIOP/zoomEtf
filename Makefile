# VARIABLES
COMPOSE = docker-compose
PROJECT_NAME = zoometf


# INIT DB (création des tables)
init-db:
	docker compose exec backend python -m app.core.init_db


# BUILD GLOBAL
build: build-backend build-frontend build-docs

rebuild: clean build

# LIFECYCLE
up:
	COMPOSE_HTTP_TIMEOUT=300 $(COMPOSE) up --build -d

down:
	$(COMPOSE) down

restart: down up

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

# BACKEND
build-backend:
	docker build -t $(PROJECT_NAME)-backend -f backend/Dockerfile .

test-backend:
	$(COMPOSE) exec backend pytest

test: test-backend

# FRONTEND
build-frontend:
	docker build -t zoometf -f frontend/Dockerfile .
	
# DOCS STATIQUES
build-docs:
	mkdir -p docs/site
	docker run --rm -v ${PWD}/docs:/data pandoc/core \
		--standalone -f markdown -t html5 ARCHITECTURE.md -o site/index.html

# CLEAN
clean:
	docker system prune -f 
	$(COMPOSE) down -v --remove-orphans

# MONITORING
monitoring:
	xdg-open http://localhost:9090 || open http://localhost:9090

grafana:
	xdg-open http://localhost:3000 || open http://localhost:3000
