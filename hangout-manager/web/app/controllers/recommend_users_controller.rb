class RecommendUsersController < ApplicationController
  before_action :set_recommend_user, only: [:show, :edit, :update, :destroy]

  # GET /recommend_users
  # GET /recommend_users.json
  def index
    @recommend_users = RecommendUser.all
  end

  # GET /recommend_users/1
  # GET /recommend_users/1.json
  def show
  end

  # GET /recommend_users/new
  def new
    @recommend_user = RecommendUser.new
    @recommend_user.user_id = current_user.id
    @recommend_user.save
    redirect_to "http://localhost:3000/get_current/#{@recommend_user.id}"
  end

  def to_current
    @recommend_user = RecommendUser.new
    @recommend_user.user_id = current_user.id
    @recommend_user.save
    @hang_outs = HangOut.all
    redirect_to "http://localhost:3000/hang_outs"
  end

  def to_current_json
    @recommend_user = RecommendUser.find(params[:id])
    @user_personal = UserPersonal.find_by(user_id: @recommend_user.user_id)
    render json: @user_personal
  end

  def get_current
    @recommend_user  = RecommendUser.find(params[:id])
    uri = URI.parse("http://recommend:5000/current/#{@recommend_user.id}")
    json = Net::HTTP.get(uri)
    @current = JSON.parse(json)
    redirect_to "http://localhost:3000/hang_outs"
  end

  # GET /recommend_users/1/edit
  def edit
  end

  # POST /recommend_users
  # POST /recommend_users.json
  def create
    @recommend_user = RecommendUser.new(recommend_user_params)
    @recommend_user.user_id = current_user.id
    @recommend_user.save
    redirect_to "http://localhost:3000/hang_outs"
  end

  # PATCH/PUT /recommend_users/1
  # PATCH/PUT /recommend_users/1.json
  def update
    respond_to do |format|
      if @recommend_user.update(recommend_user_params)
        format.html { redirect_to @recommend_user, notice: 'Recommend user was successfully updated.' }
        format.json { render :show, status: :ok, location: @recommend_user }
      else
        format.html { render :edit }
        format.json { render json: @recommend_user.errors, status: :unprocessable_entity }
      end
    end
  end

  # DELETE /recommend_users/1
  # DELETE /recommend_users/1.json
  def destroy
    @recommend_user.destroy
    respond_to do |format|
      format.html { redirect_to recommend_users_url, notice: 'Recommend user was successfully destroyed.' }
      format.json { head :no_content }
    end
  end

  private
    # Use callbacks to share common setup or constraints between actions.
    def set_recommend_user
      @recommend_user = RecommendUser.find(params[:id])
    end

    # Only allow a list of trusted parameters through.
    def recommend_user_params
      params.require(:recommend_user).permit(:user_id)
    end
end
